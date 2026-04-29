import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hashencoding import MultiResHashEncoder
from feature_blender import FeatureBlender


# --------------------------------
# Basic metrics / visualization
# --------------------------------

def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log10(x)


def to8b(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    x = np.clip(x, 0.0, 1.0)
    return (255.0 * x).astype(np.uint8)


# --------------------------------
# Positional encoding
# --------------------------------
class Embedder:
    def __init__(
        self,
        input_dims=3,
        include_input=True,
        max_freq_log2=9,
        num_freqs=10,
        log_sampling=True,
        periodic_fns=(torch.sin, torch.cos),
    ):
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.out_dim = 0
        self.embed_fns = []
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0

        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)

    def __call__(self, inputs):
        return self.embed(inputs)


def get_embedder(multires, input_dims=3, i=0):
    if i == -1:
        return lambda x: x, input_dims
    embedder_obj = Embedder(input_dims=input_dims, max_freq_log2=multires - 1, num_freqs=multires)
    return embedder_obj, embedder_obj.out_dim


# --------------------------------
# Small helpers
# --------------------------------

def _get_time_tensor(ts):
    """Best-effort conversion of ts to a tensor.

    Accepts a tensor or a list/tuple whose first element is a tensor.
    Returns a tensor with shape [..., 1] when raw time is used.
    """
    if ts is None:
        return None
    if isinstance(ts, (list, tuple)):
        if len(ts) == 0:
            return None
        ts = ts[0]
    if not torch.is_tensor(ts):
        ts = torch.tensor(ts)
    if ts.dim() == 0:
        ts = ts.view(1, 1)
    elif ts.dim() == 1:
        ts = ts.unsqueeze(-1)
    return ts


def _match_batch(t, x):
    if t is None:
        return None
    if t.shape[0] == x.shape[0]:
        return t
    if t.shape[0] == 1:
        return t.expand(x.shape[0], *t.shape[1:])
    raise ValueError(f"Cannot broadcast batch from {t.shape[0]} to {x.shape[0]}")


def _infer_embed_dim(embed_fn, input_dims=1):
    if embed_fn is None:
        return 0
    if hasattr(embed_fn, "out_dim"):
        return int(embed_fn.out_dim)
    try:
        dummy = torch.zeros(1, input_dims)
        return int(embed_fn(dummy).shape[-1])
    except Exception:
        return 0


# --------------------------------
# NeRF MLPs
# --------------------------------
class NeRFOriginal(nn.Module):
    """Standard NeRF MLP.

    This implementation is tolerant to concatenated inputs. If input_pts includes
    both position and view embeddings, the first input_ch dims are treated as xyz/pts
    features and the last input_ch_views dims can be used as view features.
    """

    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=4,
        skips=(4,),
        input_ch_views=0,
        use_viewdirs=False,
        **kwargs,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = set(skips)
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [nn.Linear(W + input_ch if i in self.skips else W, W) for i in range(1, D)]
        )
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W + input_ch_views, 3) if use_viewdirs else nn.Linear(W, 3)
        self.extra_linear = nn.Linear(W, output_ch - 4) if output_ch > 4 else None

    def forward(self, input_pts, input_views=None, ts=None):
        pts = input_pts
        views = input_views

        if pts.shape[-1] > self.input_ch:
            pts = pts[..., : self.input_ch]

        if views is None and self.use_viewdirs and input_pts.shape[-1] >= self.input_ch + self.input_ch_views:
            views = input_pts[..., -self.input_ch_views :]

        h = pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([pts, h], dim=-1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        if self.use_viewdirs:
            if views is None:
                raise ValueError("use_viewdirs=True but no view tensor was provided")
            h_rgb = torch.cat([feature, views], dim=-1)
            rgb = self.rgb_linear(h_rgb)
        else:
            rgb = self.rgb_linear(feature)

        rgb = torch.sigmoid(rgb)
        raw = torch.cat([rgb, alpha], dim=-1)
        if self.extra_linear is not None:
            raw = torch.cat([raw, self.extra_linear(h)], dim=-1)

        position_delta = torch.zeros_like(pts[..., :3])
        return raw, position_delta


class _HashTemporalNeRFBase(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=4,
        skips=(4,),
        input_ch_views=0,
        input_ch_time=1,
        use_viewdirs=False,
        embed_fn=None,
        embedtime_fn=None,
        zero_canonical=True,
        time_window_size=5,
        time_interval=1.0,
        hash_num_levels=16,
        hash_features_per_level=2,
        hash_log2_size=19,
        hash_base_resolution=16,
        hash_finest_resolution=512,
        time_hidden_dim=32,
        **kwargs,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = set(skips)
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.use_viewdirs = use_viewdirs
        self.zero_canonical = zero_canonical
        self.time_window_size = int(time_window_size)
        self.time_interval = float(time_interval)

        self.embed_fn = embed_fn
        self.embedtime_fn = embedtime_fn
        self.time_embed_dim = max(_infer_embed_dim(embedtime_fn, input_dims=1), 1)

        self.deform_mlp = nn.Sequential(
            nn.Linear(3 + self.time_embed_dim, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, 3),
        )
        nn.init.zeros_(self.deform_mlp[-1].weight)
        nn.init.zeros_(self.deform_mlp[-1].bias)

        self.hash_encoder = MultiResHashEncoder(
            num_levels=hash_num_levels,
            features_per_level=hash_features_per_level,
            log2_hashmap_size=hash_log2_size,
            base_resolution=hash_base_resolution,
            finest_resolution=hash_finest_resolution,
        )
        self.feature_blender = FeatureBlender(hash_num_levels, time_conditioned=True, time_hidden_dim=time_hidden_dim)

        trunk_in_dim = hash_features_per_level + 3 + self.time_embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
        )
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W + input_ch_views, 3) if use_viewdirs else nn.Linear(W, 3)
        self.extra_linear = nn.Linear(W, output_ch - 4) if output_ch > 4 else None

    def _prepare_inputs(self, input_pts, input_views=None):
        if input_pts.shape[-1] < 3:
            raise ValueError(f"Expected at least 3 coordinate dims, got {input_pts.shape}")
        pts = input_pts[..., :3]

        views = input_views
        if views is None and self.use_viewdirs and input_pts.shape[-1] >= 3 + self.input_ch_views:
            views = input_pts[..., -self.input_ch_views :]
        return pts, views

    def _time_context(self, t_scalar):
        if self.embedtime_fn is None:
            return t_scalar
        return self.embedtime_fn(t_scalar)

    def _deformation(self, pts, t_scalar):
        t_ctx = self._time_context(t_scalar)
        if t_ctx.dim() == 1:
            t_ctx = t_ctx.unsqueeze(-1)
        t_ctx = t_ctx[..., : self.time_embed_dim]
        if t_ctx.shape[-1] < self.time_embed_dim:
            t_ctx = F.pad(t_ctx, (0, self.time_embed_dim - t_ctx.shape[-1]))

        deform_in = torch.cat([pts, t_ctx], dim=-1)
        dx = self.deform_mlp(deform_in)
        if self.zero_canonical:
            dx = dx * t_scalar[..., :1]
        return dx, t_ctx

    def _render_from_canonical(self, canonical_pts, t_scalar, t_ctx, views=None):
        hash_feats = self.hash_encoder(canonical_pts)
        blended = self.feature_blender(hash_feats, t_scalar)
        trunk_in = torch.cat([blended, canonical_pts, t_ctx], dim=-1)
        h = self.trunk(trunk_in)
        feature = self.feature_linear(h)
        alpha = self.alpha_linear(h)

        if self.use_viewdirs:
            if views is None:
                raise ValueError("use_viewdirs=True but no view tensor was provided")
            if views.shape[-1] != self.input_ch_views:
                views = views[..., : self.input_ch_views]
            rgb = self.rgb_linear(torch.cat([feature, views], dim=-1))
        else:
            rgb = self.rgb_linear(feature)

        rgb = torch.sigmoid(rgb)
        raw = torch.cat([rgb, alpha], dim=-1)
        if self.extra_linear is not None:
            raw = torch.cat([raw, self.extra_linear(h)], dim=-1)
        return raw

    def forward(self, input_pts, input_views=None, ts=None):
        pts, views = self._prepare_inputs(input_pts, input_views)
        t = _get_time_tensor(ts)
        if t is None:
            raise ValueError("ts must not be None for temporal models")
        t = t.to(device=pts.device, dtype=pts.dtype)
        t = _match_batch(t, pts)
        t_scalar = t[..., :1]

        dx, t_ctx = self._deformation(pts, t_scalar)
        canonical_pts = pts + dx
        raw = self._render_from_canonical(canonical_pts, t_scalar, t_ctx, views=views)
        return raw, dx


class DirectTemporalNeRF(_HashTemporalNeRFBase):
    """Hash-grid canonical NeRF with time-conditioned feature blending."""

    pass


class TNeRF(DirectTemporalNeRF):
    """Compatibility alias for older code paths."""

    pass


class RecurrentTemporalNeRF(_HashTemporalNeRFBase):
    """Temporal NeRF variant using a small time-window context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ctx_dim = self.time_embed_dim * self.time_window_size
        self.window_mlp = nn.Sequential(
            nn.Linear(ctx_dim, self.W),
            nn.ReLU(inplace=True),
            nn.Linear(self.W, self.time_embed_dim),
        )

    def _time_context(self, t_scalar):
        if self.embedtime_fn is None:
            return t_scalar

        half = self.time_window_size // 2
        offsets = torch.linspace(-half, half, steps=self.time_window_size, device=t_scalar.device, dtype=t_scalar.dtype)
        contexts = []
        for off in offsets:
            t_off = t_scalar + off.view(1, 1) * self.time_interval
            emb = self.embedtime_fn(t_off)
            if emb.dim() == 1:
                emb = emb.unsqueeze(-1)
            contexts.append(emb)

        ctx = torch.cat(contexts, dim=-1)
        if ctx.shape[-1] < self.time_embed_dim * self.time_window_size:
            ctx = F.pad(ctx, (0, self.time_embed_dim * self.time_window_size - ctx.shape[-1]))
        ctx = ctx[..., : self.time_embed_dim * self.time_window_size]
        return self.window_mlp(ctx)


class NeRF(nn.Module):
    @staticmethod
    def get_by_name(name, **kwargs):
        name = name.lower()
        if name in {"nerf", "original", "nerforiginal"}:
            return NeRFOriginal(**kwargs)
        if name in {"direct_temporal", "directtemporal", "direct"}:
            return DirectTemporalNeRF(**kwargs)
        if name in {"recurrent_temporal", "recurrenttemporal", "recurrent"}:
            return RecurrentTemporalNeRF(**kwargs)
        if name in {"tnerf", "t_nerf"}:
            return TNeRF(**kwargs)
        raise NotImplementedError(f"Unknown NeRF type: {name}")


# --------------------------------
# Misc helpers
# --------------------------------

def hsv_to_rgb(hsv):
    hsv = np.asarray(hsv)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    rgb = np.zeros_like(hsv)
    conditions = [i == k for k in range(6)]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v])
    rgb[..., 1] = np.select(conditions, [t, v, v, q, p, p])
    rgb[..., 2] = np.select(conditions, [p, p, t, v, v, q])
    return rgb


def get_rays(H, W, focal, c2w):
    try:
        i, j = torch.meshgrid(
            torch.arange(W, device=c2w.device),
            torch.arange(H, device=c2w.device),
            indexing="xy",
        )
    except TypeError:
        i, j = torch.meshgrid(
            torch.arange(W, device=c2w.device),
            torch.arange(H, device=c2w.device),
        )

    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i),
        ],
        dim=-1,
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1
    )
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (
        (rays_d[..., 0] / rays_d[..., 2]) - (rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = -1.0 / (H / (2.0 * focal)) * (
        (rays_d[..., 1] / rays_d[..., 2]) - (rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], dim=-1)
    rays_d = torch.stack([d0, d1, d2], dim=-1)
    return rays_o, rays_d


def importance_sampling_coords(prob_map, n_samples, deterministic=False):
    """Sample 2D coordinates from a probability map.

    Args:
        prob_map: [H, W] or [B, H, W]
        n_samples: number of coordinates to sample
    Returns:
        coords: [n_samples, 2] or [B, n_samples, 2] in (y, x) order
    """
    if not torch.is_tensor(prob_map):
        prob_map = torch.tensor(prob_map, dtype=torch.float32)
    if prob_map.dim() == 2:
        prob_map = prob_map.unsqueeze(0)

    B, H, W = prob_map.shape
    probs = prob_map.reshape(B, -1)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    if deterministic:
        idx = torch.topk(probs, k=min(n_samples, probs.shape[-1]), dim=-1).indices
        if idx.shape[-1] < n_samples:
            pad = idx[:, -1:].expand(B, n_samples - idx.shape[-1])
            idx = torch.cat([idx, pad], dim=-1)
    else:
        idx = torch.multinomial(probs, n_samples, replacement=True)

    ys = idx // W
    xs = idx % W
    coords = torch.stack([ys, xs], dim=-1)
    return coords.squeeze(0) if coords.shape[0] == 1 else coords


def importance_sampling_ray(z_vals, weights, n_samples, deterministic=False, eps=1e-5):
    """Hierarchical importance sampling along rays.

    Args:
        z_vals: [..., N] sample locations or bin edges.
        weights: [..., N-1] or [..., N] weights over bins.
    Returns:
        new_z_samples: [..., n_samples]
    """
    if z_vals.shape[-1] == weights.shape[-1] + 1:
        bins = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
    elif z_vals.shape[-1] == weights.shape[-1]:
        bins = z_vals
    else:
        raise ValueError(
            f"z_vals and weights have incompatible shapes: {z_vals.shape} vs {weights.shape}"
        )

    w = weights + eps
    w = w / torch.sum(w, dim=-1, keepdim=True)
    cdf = torch.cumsum(w, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=z_vals.device, dtype=z_vals.dtype)
        u = u.expand(*cdf.shape[:-1], n_samples)
    else:
        u = torch.rand(*cdf.shape[:-1], n_samples, device=z_vals.device, dtype=z_vals.dtype)

    inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    cdf_g0 = torch.gather(cdf, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g0 = torch.gather(bins, -1, torch.clamp(below, max=bins.shape[-1] - 1))
    bins_g1 = torch.gather(bins, -1, torch.clamp(above, max=bins.shape[-1] - 1))

    denom = cdf_g1 - cdf_g0
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g0) / denom
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples


def depth_grad_energy(depth):
    if depth.dim() < 2:
        return torch.tensor(0.0, device=depth.device, dtype=depth.dtype)
    dx = depth[..., 1:] - depth[..., :-1]
    dy = depth[..., 1:, :] - depth[..., :-1, :]
    energy = 0.0
    if dx.numel() > 0:
        energy = energy + dx.pow(2).mean()
    if dy.numel() > 0:
        energy = energy + dy.pow(2).mean()
    return energy
