
import torch
import torch.nn as nn
import numpy as np

class MultiResHashEncoder(nn.Module):
    def __init__(self,
                 num_levels=12,
                 features_per_level=2,
                 log2_hashmap_size=17,
                 base_resolution=16,
                 finest_resolution=256):
        super().__init__()

        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size

        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        self.b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (num_levels - 1))

        self.tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, features_per_level)
            for _ in range(num_levels)
        ])

    def hash(self, coords):
        x, y, z = coords[...,0], coords[...,1], coords[...,2]
        return (x * 73856093 ^ y * 19349663 ^ z * 83492791) % self.hashmap_size

    def forward(self, x):
        feats_all = []

        for lvl in range(self.num_levels):
            resolution = int(self.base_resolution * (self.b ** lvl))

            x_scaled = x * resolution
            x0 = torch.floor(x_scaled).long()
            w = x_scaled - x0.float()

            feat = 0

            for ix in [0,1]:
                for iy in [0,1]:
                    for iz in [0,1]:
                        corner = torch.stack([
                            x0[...,0] + ix,
                            x0[...,1] + iy,
                            x0[...,2] + iz
                        ], dim=-1)

                        idx = self.hash(corner)
                        f = self.tables[lvl](idx)

                        wx = w[...,0] if ix else (1-w[...,0])
                        wy = w[...,1] if iy else (1-w[...,1])
                        wz = w[...,2] if iz else (1-w[...,2])

                        weight = (wx * wy * wz).unsqueeze(-1)
                        feat = feat + weight * f

            feats_all.append(feat)

        return feats_all
