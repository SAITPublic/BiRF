#!/bin/bash

read -a gpus

for gpu in "${gpus[*]}"
do
    for Bike Lifestyle Palace Robot Spaceship Steamtrain Toad Wineholder
    do
        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 2 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 2 --seed 2023

        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 2 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 2 --seed 2023
    done
done
