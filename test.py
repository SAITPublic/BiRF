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

import argparse
import math
import os
import time
import json

import gin
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from typing import *
from datetime import datetime

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from lib.models.ngp import NGPradianceField
from lib.utils import render_image, set_random_seed, load_dataset, load_occgrid, load_model

from nerfacc import ContractionType, OccupancyGrid


class ExtendAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.register('action', 'extend', ExtendAction)
    parser.add_argument(
        "configs",
        action="append",
        help="path to config files",
    )
    parser.add_argument(
        "--bind",
        nargs='+',
        action="extend",
        help="param to bind",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        choices=[
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
            # nsvf synthetic
            "Bike",
            "Lifestyle",
            "Palace",
            "Robot",
            "Spaceship",
            "Steamtrain",
            "Toad",
            "Wineholder",
            # nsvf TankAndTemple
            "Barn", 
            "Caterpillar", 
            "Family", 
            "Ignatius", 
            "Truck",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=2,
        help="number of features"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed number"
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="path for checkpoint directory"
    )

    return parser.parse_args()


@gin.configurable
def main(
    scene: str,
    ckpt_dir: str,
    n_features: int=2,
    seed: int = 2023,
    log_dir: str = "./logs",
    prefix: Optional[str] = None,
    postfix: Optional[str] = None,
    max_steps: int = 20000,
    render_n_samples: int = 1024,
    test_chunk_size: int = 16384,
    aabb: List[float] = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
    data_root_fp: str = "data/nerf_synthetic/",
    train_split: str = "train",
    cone_angle: float = 0.0,
    sparsity_weight: float = 2e-5,
    render_per_frame: int = -1,
):
    # log
    save_path = f"{log_dir}/{scene}" if ckpt_dir == None else ckpt_dir

    if prefix is not None:
        save_path = f"{prefix}_{save_path}"
    if postfix is not None:
        save_path = f"{save_path}_{postfix}"
    
    save_path = f"{save_path}_{n_features}"

    print(f'Evaluation for pretrained model in "{save_path}"')
    results = {}

    # setup the dataset
    test_dataset_kwargs = {}

    target_sample_batch_size = 1 << 18
    grid_resolution = 128

    test_dataset, data_root_fp = load_dataset(
        scene=scene,
        data_root_fp=data_root_fp,
        split="test",
        num_rays=None,
        dataset_kwargs=test_dataset_kwargs,
        device=device,
    )
    
    if os.path.exists(os.path.join(f"{data_root_fp}", str(scene), "bbox.txt")):
        aabb = list(np.loadtxt(os.path.join(f"{data_root_fp}", str(scene), "bbox.txt"))[:6])

    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()
    alpha_thre = 0

    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    
    radiance_field = NGPradianceField(
        aabb=aabb,
        n_features_per_level=n_features,
    ).to(device)
    radiance_field = load_model(radiance_field, save_path, device=device)

    occupancy_grid = OccupancyGrid(
        roi_aabb=aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    occupancy_grid = load_occgrid(occupancy_grid, save_path, device=device, res=grid_resolution)

    # metrics
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)


    radiance_field = radiance_field.half()
    if render_per_frame > 0:
        os.makedirs(f"{save_path}/imgs", exist_ok=True)
    # evaluation
    init = time.time()
    radiance_field.eval()
    psnr_list, ssim_list, lpips_list = [], [], []
    with torch.no_grad():
        for j in tqdm.tqdm(range(len(test_dataset))):
            data = test_dataset[j]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]
            # rendering
            rgb, acc, depth, _ = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
                # test options
                test_chunk_size=test_chunk_size,
            )
            if render_per_frame > 0 and j % render_per_frame == 0:
                imageio.imwrite(
                    f"{save_path}/imgs/{j:03d}.png",
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )
            
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            
            rgb = rgb.permute(-1, 0, 1)[None, ...]
            pixels = pixels.permute(-1, 0, 1)[None, ...]
            ssim = SSIM(rgb, pixels)
            lpips = LPIPS(rgb, pixels)
            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            lpips_list.append(lpips.item())
    
    
    psnr_avg = sum(psnr_list) / len(psnr_list)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    lpips_avg = sum(lpips_list) / len(lpips_list)
    print(f"Evaluation PSNR: {round(psnr_avg, 2):.2f}")
    print(f"Evaluation SSIM: {round(ssim_avg, 3):.3f}")
    print(f"Evaluation LPIPS: {round(lpips_avg, 3):.3f}")
    
    test_time = time.time() - init
    
    render_speed = len(test_dataset) / test_time
    encoding_size = os.path.getsize(f"{save_path}/encoding.npz")
    network_size = os.path.getsize(f"{save_path}/network.ckpt")
    occgrid_size = os.path.getsize(f"{save_path}/occgrid.npz")
    total_size = encoding_size + network_size + occgrid_size
    print(f"Evaluation encoding size: {round((encoding_size / 2 ** 20), 2):.2f} MB")
    print(f"Evaluation network size: {round((network_size / 2 ** 20), 2):.2f} MB")
    print(f"Evaluation occgrid size: {round((occgrid_size / 2 ** 20), 2):.2f} MB")
    print(f"Evaluation total size: {round((total_size / 2 ** 20), 2):.2f} MB")
    results["psnr"] = round(psnr_avg, 2)
    results["ssim"] = round(ssim_avg, 3)
    results["lpips"] = round(lpips_avg, 3)
    results["test_time"] = round(test_time, 2)
    results["render_speed"] = round(render_speed, 2)
    results['size'] = round(total_size / 2 ** 20, 2)
    
    with open(f"{save_path}/results.json", 'w') as f:
        json.dump(results, f)
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())

    print("Evaluation done")
    
    return
            

if __name__ == "__main__":
    device = "cuda:0"
    args = parse_args()
    set_random_seed(args.seed)

    print(f"Radom seed number: {args.seed}")
    print(f"Configuration files: {args.configs}")
    print(f"Binding parameters: {args.bind}")
    gin.parse_config_files_and_bindings(args.configs, args.bind)

    main(ckpt_dir=args.ckpt_dir, scene=args.scene, n_features=args.n_features)
