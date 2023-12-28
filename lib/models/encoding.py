"""
"Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)"
"""

"""
Modified from NerfAcc (https://github.com/KAIR-BAIR/nerfacc)
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import gin
import torch
import numpy as np
import torch.nn as nn
import tinycudann as tcnn
import math
from typing import *


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = False):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)

        return latent


class HybridEncoding(nn.Module):
    def __init__(
        self, 
        n_input_dims, 
        grid_encoding_config,
        plane_encoding_config,
    ) -> None:
        super().__init__()

        self.grid_coef_0 = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=grid_encoding_config
        )
        self.plane_coef_0 = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=plane_encoding_config
        )
        self.plane_coef_1 = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=plane_encoding_config
        )
        self.plane_coef_2 = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=plane_encoding_config
        )
        self.n_output_dims = sum([
            self.grid_coef_0.n_output_dims,
            self.plane_coef_0.n_output_dims,
            self.plane_coef_1.n_output_dims,
            self.plane_coef_2.n_output_dims,
        ])

    def forward(self, in_tensor):
        features = torch.cat(
            [
                self.grid_coef_0(in_tensor[..., [0, 1, 2]]),
                self.plane_coef_0(in_tensor[..., [0, 1]]),
                self.plane_coef_1(in_tensor[..., [1, 2]]),
                self.plane_coef_2(in_tensor[..., [2, 0]]),
            ], dim=-1
        ) # [4 * N, n_levels * n_features_per_level]

        return features


@gin.configurable
class NetworkWithInputEncoding(nn.Module):
    def __init__(
        self,
        n_input_dims: int = 3,
        n_output_dims: int = 1 + 15,
        network_config: Dict = None,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        n_levels: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 1024,
        interpolation: str = 'BinaryLinear',
        pos_deg: int = 4,
    ):
        super().__init__()

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        # 3D feature grid
        grid_encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": 16,
            "per_level_scale": per_level_scale,
            "interpolation": interpolation,
        }

        # 2D feature grid
        plane_encoding_config = {
            "otype": "HashGrid",
            "n_levels": 4,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size - 2,
            "base_resolution": max_resolution // 4,
            "per_level_scale": 2.0,
            "interpolation": interpolation,
        }

        self.encoding = HybridEncoding(
            n_input_dims,
            grid_encoding_config=grid_encoding_config,
            plane_encoding_config=plane_encoding_config,
        )

        # positional encoding
        network_output_dims = self.encoding.n_output_dims

        self.posi_encoder = SinusoidalEncoder(3, 0, pos_deg, False) if pos_deg > 0 else None
        network_output_dims += self.posi_encoder.latent_dim if pos_deg > 0 else 0

        self.network = tcnn.Network(
            n_input_dims=network_output_dims,
            n_output_dims=n_output_dims,
            network_config=network_config
        )
        
    def forward(self, in_tensor):
        x = self.encoding(in_tensor)
        if self.posi_encoder is not None:
            x = torch.cat([x, self.posi_encoder(in_tensor)], dim=-1)
        return self.network(x)
