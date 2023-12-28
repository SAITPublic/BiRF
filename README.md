# Binary Radiance Fields (NeurIPS 2023)

<!-- <p align="left">
    <a href="https://arxiv.org/abs/2306.07581"><img src="https://img.shields.io/badge/arxiv-2306.07581-b31b1b"></a>
    <a href="https://seungjooshin.github.io/BiRF"><img src="https://img.shields.io/badge/githubpages-BiRF-222222"></a>
</p> -->


### [Paper](https://arxiv.org/abs/2306.07581) | [Project Page](https://seungjooshin.github.io/BiRF/)

This repository contains the code release for the paper: 
<!-- This is the official implementation of [Binary Radiance Fields](https://arxiv.org/abs/2306.07581): -->

> **Binary Radiance Fields** \
> [Seungjoo Shin](https://seungjooshin.github.io)<sup>1</sup>, and [Jaesik Park](https://jaesik.info)<sup>2</sup> \
> <sup>1</sup> POSTECH, <sup>2</sup> Seoul National University \
> *Conference on Neural Information Processing Systems (**NeurIPS**)*, *New Orleans*, 2023

<div style="text-align:center">
    <img src="./assets/overview.png"/>
</div>

## Setup

We have tested on ```PyTorch==1.13.0``` with ```CUDA==11.7```.

### Clone the repository:

```bash
git clone https://github.com/SAITPublic/BiRF.git
cd BiRF
```

### Create a environment:

``` bash
conda create --name birf -y python=3.8
conda activate birf
```

### Install packages:

``` bash
# install PyTorch==1.13.0 with CUDA==11.7
conda install pytorch==1.13.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.7 -c pytorch

# install custom tiny-cuda-nn
pip install git+https://github.com/seungjooshin/tiny-cuda-nn/@bit#subdirectory=bindings/torch

# install requirements
pip install -r requirements.txt
```

### Prepare datasets:

We support three datasets for evaluation.

- [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
- [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
- [Tanks and Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)

By default, we put the dataset under ```./data/``` as:

```
├── data
    ├── nerf_synthetic
      ├── chair
      ├── drums
      ├── ....
    ├── Synthetic_NSVF
      ├── Bike
      ├── Liferstyle
      ├── ....
    ├── TanksAndTemples
      ├── Barn
      ├── Caterpillar
      ├── ....
```

## Running



### Training:
``` bash
# python train.py ./config/{size}.gin --scene {scene} --n_features {n_features}
python train.py ./config/base.gin --scene chair --n_features 2
```
- `size`: the size of hash table
- `scene`: the scene to reconstruct
- `n_features`: the number of features
- The result is saved as `{log_dir}/results_{seed}.json`.

### Testing:


``` bash
# python test.py ./config/{size}.gin --scene {scene} --n_features {n_features} --log_dir {path_to_log_dir}
python test.py ./config/base.gin --scene chair --n_features 2 --log_dir ./logs/chair_f2_2023
```
- `size`: the size of hash table
- `scene`: the scene to reconstruct
- `n_features`: the number of features
- `log_dir`: the path to log directory
- The result is saved as `{log_dir}/results.json`.

By default, we save the log under ```./logs/```.
```
├── logs
    ├── chair_b_2
      ├── imgs
        ├── 0000.png
        ├── 0001.png
        ├── ....
      ├── config.gin
      ├── encoding.npz
      ├── network.ckpt
      ├── occgrid.npz
      ├── results.json
    ├── chair_s_2
    ├── ....
```
## Citation

If you find our code or paper useful, please consider citing our paper:
```BibTeX
@inproceedings{
  shin2023binary,
  title={Binary Radiance Fields},
  author={Seungjoo Shin and Jaesik Park},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
