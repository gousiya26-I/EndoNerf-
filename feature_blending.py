import torch
import torch.nn as nn


class FeatureBlender(nn.Module):
    """Blend per-level hash features.

    If `time_conditioned=True`, the level weights are modulated by time `t`.
    Otherwise, a single global softmax over learnable level logits is used.
    """

    def __init__(self, num_levels, time_conditioned=False, time_hidden_dim=32):
        super().__init__()
        self.num_levels = num_levels
        self.time_conditioned = time_conditioned

        # Start uniform after softmax.
        self.level_logits = nn.Parameter(torch.zeros(num_levels))

        if time_conditioned:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(time_hidden_dim, num_levels),
            )

    def forward(self, features, t=None):
        """
        Args:
            features: Tensor [..., num_levels, feat_dim] or list/tuple of per-level tensors.
            t: Optional time tensor [..., 1]. Used only if time_conditioned=True.

        Returns:
            Tensor [..., feat_dim]
        """
        if isinstance(features, (list, tuple)):
            features = torch.stack(features, dim=-2)

        if features.dim() < 3:
            raise ValueError(
                f"Expected features with at least 3 dims [..., num_levels, feat_dim], got {features.shape}"
            )

        if features.shape[-2] != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, got {features.shape[-2]}"
            )

        if self.time_conditioned and t is not None:
            t = t[..., :1]
            time_logits = self.time_mlp(t)  # [..., num_levels]
            base = self.level_logits.view(*([1] * (time_logits.dim() - 1)), self.num_levels)
            logits = base + time_logits
            weights = torch.softmax(logits, dim=-1)  # [..., num_levels]
            weights = weights.unsqueeze(-1)  # [..., num_levels, 1]
        else:
            weights = torch.softmax(self.level_logits, dim=0)
            weights = weights.view(*([1] * (features.dim() - 2)), self.num_levels, 1)

        return torch.sum(features * weights, dim=-2)
