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
from lib.utils import render_image, set_random_seed, load_dataset, save_occgrid, save_model

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

    return parser.parse_args()


@gin.configurable
def main(
    scene: str,
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
    os.makedirs(log_dir, exist_ok=True)
    save_path = f"{log_dir}/{scene}"

    if prefix is not None:
        save_path = f"{prefix}_{save_path}"
    if postfix is not None:
        save_path = f"{save_path}_{postfix}"
    
    save_path = f"{save_path}_{n_features}"
    os.makedirs(f"{save_path}", exist_ok=True)

    if len([f for f in os.listdir(save_path) if f.startswith('results')]) > 0:
        print(f"Already done in {save_path}")
        exit()
    results = {}

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}

    target_sample_batch_size = 1 << 18
    grid_resolution = 128

    train_dataset, data_root_fp = load_dataset(
        scene=scene,
        data_root_fp=data_root_fp,
        split=train_split,
        num_rays=render_n_samples,
        dataset_kwargs=train_dataset_kwargs,
        device=device,
    )

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

    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=1e-2, eps=1e-15,
    )

    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=1000
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )

    occupancy_grid = OccupancyGrid(
        roi_aabb=aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)

    # metrics
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    # training
    step = 0
    init = time.time()
    pbar = tqdm.tqdm(range(max_steps + 1))
    psnrs = []
    psnr_avg = 0
    for step in pbar:
        pbar.set_postfix({'steps' : step})
        if step > 0 and len(psnrs) > 0:
            if step % 100 == 0:
                psnr_avg = sum(psnrs) / len(psnrs)
                psnrs = []
            pbar.set_postfix({'steps' : step, 'train psnr': psnr_avg})
        
        radiance_field.train()
        
        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        
        def occ_eval_fn(x):
            if cone_angle > 0.0:
                # randomly sample a camera for computing step size.
                camera_ids = torch.randint(
                    0, len(train_dataset), (x.shape[0],), device=device
                )
                origins = train_dataset.camtoworlds[camera_ids, :3, -1]
                t = (origins - x).norm(dim=-1, keepdim=True)
                # compute actual step size used in marching, based on the distance to the camera.
                step_size = torch.clamp(
                    t * cone_angle, min=render_step_size
                )
                # filter out the points that are not in the near far plane.
                if (near_plane is not None) and (far_plane is not None):
                    step_size = torch.where(
                        (t > near_plane) & (t < far_plane),
                        step_size,
                        torch.zeros_like(step_size),
                    )
            else:
                step_size = render_step_size
            # compute occupancy
            density = radiance_field.query_density(x)
            return density * step_size

        # update occupancy grid
        occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

        # render
        rgb, acc, depth, n_rendering_samples = render_image(
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
        )
        if n_rendering_samples == 0:
            continue

        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays
            * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)
        alive_ray_mask = acc.squeeze(-1) > 0
        loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask]) + sparsity_weight * radiance_field.sparsity

        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        psnrs.append(psnr.item())

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step == max_steps and step > 0:
            print("Training done")
            radiance_field = radiance_field.half()
            save_model(radiance_field.state_dict(), save_path)
            save_occgrid(occupancy_grid, save_path)

            if render_per_frame > 0:
                os.makedirs(f"{save_path}/imgs", exist_ok=True)

            # evaluation
            elapsed_time = time.time() - init
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
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    if render_per_frame > 0 and j % render_per_frame == 0:
                        imageio.imwrite(
                            f"{save_path}/imgs/{j:03d}.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                    
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

            test_time = time.time() - elapsed_time - init
            init += test_time
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
            results["train_time"] = round(elapsed_time, 2)
            results["test_time"] = round(test_time, 2)
            results["render_speed"] = round(render_speed, 2)
            results['size'] = round(total_size / 2 ** 20, 2)
            
            with open(f"{save_path}/results_{seed:04d}.json", 'w') as f:
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

    main(seed=args.seed, scene=args.scene, n_features=args.n_features)
