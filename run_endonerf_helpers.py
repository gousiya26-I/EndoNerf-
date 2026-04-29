from Hash_encoding import MultiResHashEncoder
from feature_blending import FeatureBlender

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.autograd.set_detect_anomaly(True)
searchsorted = torch.searchsorted


# -------------------------
# Misc utilities
# -------------------------
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log10(x)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# -------------------------
# Positional encoding
# -------------------------
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0

        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        n_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=n_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=n_freqs)

        # Convert to Python floats so the embedder works on both CPU and GPU.
        for freq in freq_bands:
            freq = float(freq.item()) if torch.is_tensor(freq) else float(freq)
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims

    embedder_obj = Embedder(
        include_input=True,
        input_dims=input_dims,
        max_freq_log2=multires - 1,
        num_freqs=multires,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    )
    return lambda x, eo=embedder_obj: eo.embed(x), embedder_obj.out_dim


# -------------------------
# Small helpers
# -------------------------
def _get_time_tensor(ts):
    if isinstance(ts, (list, tuple)):
        ts = ts[0]
    if ts is None:
        return None
    if not torch.is_tensor(ts):
        ts = torch.tensor(ts, dtype=torch.float32)
    if ts.dim() == 0:
        ts = ts.view(1, 1)
    elif ts.dim() == 1:
        ts = ts.unsqueeze(-1)
    return ts


def _match_batch(t, ref):
    if t is None:
        return None
    if t.shape[0] == 1 and ref.shape[0] > 1:
        t = t.expand(ref.shape[0], -1)
    return t


def _infer_embed_dim(embed_fn, input_dim=3):
    if embed_fn is None:
        return input_dim
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim)
            out = embed_fn(dummy)
        return out.shape[-1]
    except Exception:
        return input_dim


# -------------------------
# Basic NeRF MLP
# -------------------------
class NeRFOriginal(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        input_ch_time=1,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        memory=None,
        embed_fn=None,
        embedtime_fn=None,
        output_color_ch=3,
        zero_canonical=True,
        time_window_size=1,
        time_interval=0.0,
        **kwargs,
    ):
        super().__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = list(skips)
        self.use_viewdirs = use_viewdirs

        memory = [] if memory is None else list(memory)
        self.memory = memory
        self.embed_fn = embed_fn
        self.embedtime_fn = embedtime_fn
        self.output_ch = output_ch
        self.output_color_ch = output_color_ch
        self.zero_canonical = zero_canonical
        self.time_window_size = time_window_size
        self.time_interval = time_interval

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError("memory connections are not implemented in this cleaned version")

            in_channels = W + input_ch if i in self.skips else W
            layers.append(nn.Linear(in_channels, W))

        self.pts_linears = nn.ModuleList(layers)

        # Official NeRF-style view branch.
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, output_color_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, ts=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)

            for l in self.views_linears:
                h = F.relu(l(h))

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs, torch.zeros_like(input_pts[:, :3])

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# -------------------------
# Direct temporal model: canonical -> hash grid -> blend -> NeRF MLP
# -------------------------
class DirectTemporalNeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        input_ch_time=1,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        embed_fn=None,
        zero_canonical=True,
        temporal_blending=True,
        hash_num_levels=12,
        hash_features_per_level=2,
        log2_hashmap_size=17,
        base_resolution=16,
        finest_resolution=256,
    ):
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = list(skips)
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical
        self.scale = 1.5

        self.pos_embed_dim = _infer_embed_dim(embed_fn, input_ch)

        self.hash_encoder = MultiResHashEncoder(
            num_levels=hash_num_levels,
            features_per_level=hash_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )
        self.blender = FeatureBlender(
            self.hash_encoder.num_levels,
            time_conditioned=temporal_blending,
        )

        self.hash_dim = self.hash_encoder.features_per_level

        self._occ = NeRFOriginal(
            D=D,
            W=W,
            input_ch=self.pos_embed_dim + self.hash_dim,
            input_ch_views=input_ch_views,
            output_ch=output_ch,
            skips=skips,
            use_viewdirs=use_viewdirs,
        )

        self._time, self._time_out = self.create_time_net()
        nn.init.zeros_(self._time_out.weight)
        nn.init.zeros_(self._time_out.bias)

    def create_time_net(self):
        hidden = self._occ.pts_linears[0].out_features
        layers = [nn.Linear(self.pos_embed_dim + self.input_ch_time, hidden)]
        for i in range(len(self._occ.pts_linears) - 1):
            in_ch = hidden
            if i in self.skips:
                in_ch += self.pos_embed_dim
            layers.append(nn.Linear(in_ch, hidden))
        return nn.ModuleList(layers), nn.Linear(hidden, 3)

    def query_time(self, pts_encoded, t):
        h = torch.cat([pts_encoded, t], dim=-1)
        for i, l in enumerate(self._time):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([pts_encoded, h], dim=-1)
        return self._time_out(h)

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        t = _get_time_tensor(ts)
        if t is None:
            raise ValueError("ts must not be None for DirectTemporalNeRF")
        t = t.to(input_pts.device)
        t = _match_batch(t, input_pts)

        input_pts_orig = input_pts[..., :3]
        pts_encoded = self.embed_fn(input_pts_orig) if self.embed_fn is not None else input_pts_orig

        is_zero_time = torch.allclose(t, torch.zeros_like(t))
        if is_zero_time and self.zero_canonical:
            dx = torch.zeros_like(input_pts_orig)
        else:
            dx = self.query_time(pts_encoded, t)

        canonical_pts = input_pts_orig + dx
        canonical_pts = canonical_pts / self.scale

        multi_feats = self.hash_encoder(canonical_pts)
        blended_feats = self.blender(multi_feats, t)

        pos_feats = self.embed_fn(canonical_pts) if self.embed_fn is not None else canonical_pts
        input_pts_final = torch.cat([pos_feats, blended_feats], dim=-1)

        out, _ = self._occ(torch.cat([input_pts_final, input_views], dim=-1), ts)
        return out, dx


# -------------------------
# Temporal models used by older code paths
# -------------------------
class TNeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        input_ch_time=1,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        memory=None,
        embed_fn=None,
        embedtime_fn=None,
        zero_canonical=True,
        time_window_size=1,
        time_interval=0.0,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = list(skips)
        self.use_viewdirs = use_viewdirs
        self.memory = [] if memory is None else list(memory)
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical

        self._occ = NeRFOriginal(
            D=D,
            W=W,
            input_ch=input_ch + input_ch_time,
            input_ch_views=input_ch_views,
            input_ch_time=input_ch_time,
            output_ch=output_ch,
            skips=skips,
            use_viewdirs=use_viewdirs,
            memory=memory,
            embed_fn=embed_fn,
            embedtime_fn=embedtime_fn,
        )

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        t = _get_time_tensor(ts)
        if t is None:
            raise ValueError("ts must not be None for TNeRF")
        t = t.to(input_pts.device)
        t = _match_batch(t, input_pts)

        assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time"
        return self._occ(torch.cat([input_pts, t, input_views], dim=-1), t)


class RecurrentTemporalNeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        input_ch_time=1,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        memory=None,
        embed_fn=None,
        embedtime_fn=None,
        zero_canonical=True,
        time_window_size=1,
        time_interval=0.0,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = list(skips)
        self.use_viewdirs = use_viewdirs
        self.memory = [] if memory is None else list(memory)
        self.embed_fn = embed_fn
        self.embedtime_fn = embedtime_fn if embedtime_fn is not None else (lambda x: x)
        self.zero_canonical = zero_canonical
        self.time_window_size = time_window_size
        self.time_interval = time_interval

        self.pos_embed_dim = _infer_embed_dim(embed_fn, input_ch)
        self.time_embed_dim = _infer_embed_dim(self.embedtime_fn, input_ch_time)

        self._occ = NeRFOriginal(
            D=D,
            W=W,
            input_ch=self.pos_embed_dim,
            input_ch_views=input_ch_views,
            input_ch_time=input_ch_time,
            output_ch=output_ch,
            skips=skips,
            use_viewdirs=use_viewdirs,
            memory=memory,
            embed_fn=embed_fn,
            embedtime_fn=embedtime_fn,
        )
        self._time_hidden, self._time_gru, self._time_out = self.create_time_net()
        nn.init.zeros_(self._time_out[-1].weight)
        nn.init.zeros_(self._time_out[-1].bias)

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.time_embed_dim, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError

            in_channels = self.W + self.input_ch if i in self.skips else self.W
            layers.append(nn.Linear(in_channels, self.W))

        if self.time_window_size > 1:
            gru = nn.GRU(input_size=self.W, hidden_size=self.W, num_layers=self.time_window_size - 1)
        else:
            gru = None

        out = nn.Sequential(
            nn.Linear(self.W, self.W // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.W // 2, 3),
        )
        return nn.ModuleList(layers), gru, out

    def query_time_hidden(self, new_pts, t, net):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([new_pts, h], dim=-1)
        return h

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        t = _get_time_tensor(ts)
        if t is None:
            raise ValueError("ts must not be None for RecurrentTemporalNeRF")
        t = t.to(input_pts.device)
        t = _match_batch(t, input_pts)

        assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = float(t[0, 0].item())

        input_pts_orig = input_pts[:, :3]

        if cur_time == 0.0 and self.zero_canonical:
            dx = torch.zeros_like(input_pts_orig)
        else:
            if self.time_window_size > 1 and self._time_gru is not None:
                time_hidden_window = []
                for i in range(1, self.time_window_size):
                    prev_t = torch.maximum(
                        torch.zeros_like(t[:, :1]),
                        t[:, :1] - i * self.time_interval,
                    )
                    prev_t_emb = self.embedtime_fn(prev_t)
                    time_hidden_window.append(
                        self.query_time_hidden(input_pts_orig, prev_t_emb, self._time_hidden)
                    )

                time_hidden = torch.stack(time_hidden_window)
                curr_time_hidden = self.query_time_hidden(
                    input_pts_orig,
                    self.embedtime_fn(t[:, :1]),
                    self._time_hidden,
                ).unsqueeze(0)
                out_h, _ = self._time_gru(curr_time_hidden, time_hidden)
                out_h = out_h.squeeze(0)
            else:
                out_h = self.query_time_hidden(
                    input_pts_orig,
                    self.embedtime_fn(t[:, :1]),
                    self._time_hidden,
                )

            dx = self._time_out(out_h)

        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts_orig + dx)
        else:
            input_pts = input_pts_orig + dx

        out, _ = self._occ(torch.cat([input_pts, input_views], dim=-1), t)
        return out, dx


# -------------------------
# Factory
# -------------------------
class NeRF:
    @staticmethod
    def get_by_name(type, *args, **kwargs):
        print("NeRF type selected: %s" % type)

        if type == "original":
            model = NeRFOriginal(*args, **kwargs)
        elif type == "direct_temporal":
            model = DirectTemporalNeRF(*args, **kwargs)
        elif type == "recurrent_temporal":
            model = RecurrentTemporalNeRF(*args, **kwargs)
        elif type == "tnerf":
            model = TNeRF(*args, **kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model


# -------------------------
# Helpers
# -------------------------
def hsv_to_rgb(h, s, v):
    """
    h,s,v in range [0,1]
    """
    hi = torch.floor(h * 6)
    f = h * 6.0 - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    rgb = torch.cat([hi, hi, hi], -1) % 6
    rgb[rgb == 0] = torch.cat((v, t, p), -1)[rgb == 0]
    rgb[rgb == 1] = torch.cat((q, v, p), -1)[rgb == 1]
    rgb[rgb == 2] = torch.cat((p, v, t), -1)[rgb == 2]
    rgb[rgb == 3] = torch.cat((p, q, v), -1)[rgb == 3]
    rgb[rgb == 4] = torch.cat((t, p, v), -1)[rgb == 4]
    rgb[rgb == 5] = torch.cat((v, p, q), -1)[rgb == 5]
    return rgb


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / (rays_d[..., 2] + 1e-6)
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / (rays_o[..., 2] + 1e-6)
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / (rays_o[..., 2] + 1e-6)
    o2 = 1.0 + 2.0 * near / (rays_o[..., 2] + 1e-6)

    d0 = -1.0 / (W / (2.0 * focal)) * (
        rays_d[..., 0] / (rays_d[..., 2] + 1e-6) - rays_o[..., 0] / (rays_o[..., 2] + 1e-6)
    )
    d1 = -1.0 / (H / (2.0 * focal)) * (
        rays_d[..., 1] / (rays_d[..., 2] + 1e-6) - rays_o[..., 1] / (rays_o[..., 2] + 1e-6)
    )
    d2 = -2.0 * near / (rays_o[..., 2] + 1e-6)

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def importance_sampling_coords(weights, N_samples, det=False, pytest=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    return inds, u, cdf


# Hierarchical sampling (section 5.2)
def importance_sampling_ray(bins, weights, N_samples, det=False, pytest=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def ray_sampling_importance_from_masks(masks):
    freq = (1.0 - masks).sum(0)
    p = freq / torch.sqrt((torch.pow(freq, 2)).sum())
    return masks * (1.0 + p)


grad_kernel_x = torch.Tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0, -1.0]])
grad_kernel_y = torch.Tensor([[1.0, 2.0, 1.0], [0, 0, 0], [-1.0, -2.0, -1.0]])


def depth_grad_energy(x, step=1, K=50.0):
    kernel_x = grad_kernel_x.view((1, 1, 3, 3)).to(x.device)
    kernel_y = grad_kernel_y.view((1, 1, 3, 3)).to(x.device)

    G_x = x
    for _ in range(step):
        G_x = F.conv2d(G_x, kernel_x, padding=1)

    G_y = x
    for _ in range(step):
        G_y = F.conv2d(G_y, kernel_y, padding=1)

    G = torch.pow(G_x, 2) + torch.pow(G_y, 2)
    G = G.squeeze(1).reshape([x.shape[0], -1])
    E = torch.sum(1.0 - torch.exp(-G / (K**2)), dim=1)

    return E
