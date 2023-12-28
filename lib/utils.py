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

import random
from typing import *

import numpy as np
import torch
from lib.datasets.utils import Rays, namedtuple_map

from nerfacc import OccupancyGrid, ray_marching, rendering

def load_dataset(
    scene: str,
    data_root_fp: str,
    split: str,
    num_rays: Optional[int],
    dataset_kwargs: Dict,
    device: str,
):
    if scene in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
        from lib.datasets.nerf_synthetic import SubjectLoader
        data_root_fp = 'data/nerf_synthetic/'
    elif scene in ["Bike", "Lifestyle", "Palace", "Robot", "Spaceship", "Steamtrain", "Toad", "Wineholder"]:
        from lib.datasets.nsvf import SubjectLoader
        data_root_fp = 'data/Synthetic_NSVF/'
    elif scene in ["Barn", "Caterpillar", "Family", "Ignatius", "Truck"]:
        from lib.datasets.tanksandtemple import SubjectLoader
        data_root_fp = 'data/TanksAndTemple/'

    dataset = SubjectLoader(
        subject_id=scene,
        root_fp=data_root_fp,
        split=split,
        num_rays=num_rays,
        **dataset_kwargs,
    )

    dataset.images = dataset.images.to(device)
    dataset.camtoworlds = dataset.camtoworlds.to(device)
    dataset.K = dataset.K.to(device)

    return dataset, data_root_fp


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )


def save_model(state_dict, save_path):
    encoding_params = dict()
    params_keys = [key for key in state_dict.keys() if key.startswith('mlp_base.encoding') ]
    for key in params_keys:
        params = np.array(state_dict[key].cpu() >=0).astype(np.bool_)
        encoding_params[key] = np.packbits(params)
        del state_dict[key]
    
    np.savez_compressed(f'{save_path}/encoding.npz', **encoding_params)
    torch.save(state_dict, f'{save_path}/network.ckpt')

    print(f"Encoding parameters are saved as {save_path}/encoding.npz")
    print(f"Network parameters are saved as {save_path}/network.ckpt")


def load_model(radiance_field, save_path, device):
    radiance_field.load_state_dict(torch.load(f"{save_path}/network.ckpt"), strict=False)
    encoding_params = np.load(f"{save_path}/encoding.npz")
    
    params_keys = [key for key in radiance_field.state_dict().keys() if key.startswith('mlp_base.encoding')]
    model_params = {}
    for key in params_keys:
        num = radiance_field.state_dict()[key].shape[0]
        params = np.unpackbits(encoding_params[key]).astype(np.float16)[:num]
        model_params[key] = torch.tensor(2 * params - 1).to(device)

    radiance_field.load_state_dict(model_params, strict=False)

    return radiance_field


def save_occgrid(occupancy_grid, save_path):
    binary = np.array(occupancy_grid._binary.cpu().flatten()).astype(np.bool_)
    data = np.packbits(binary)
    np.savez_compressed(f"{save_path}/occgrid.npz", data=data)
    print(f"Occupancy grid is saved as {save_path}/occgrid.npz")


def load_occgrid(occupancy_grid, save_path, device, res=128):
    data = np.load(f"{save_path}/occgrid.npz")['data']
    binary = np.unpackbits(data).reshape(res, res, res)
    binary = torch.tensor(binary).type(torch.bool).to(device)
    occupancy_grid._binary = binary
    
    return occupancy_grid