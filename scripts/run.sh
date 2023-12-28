#!/bin/bash

read -a gpus

# chair drums ficus hotdog lego materials mic ship
# Bike Lifestyle Palace Robot Spaceship Steamtrain Toad Wineholder
# Barn Caterpillar Family Ignatius Truck

for gpu in "${gpus[*]}"
do
    for scene in ficus hotdog lego materials mic ship Lifestyle Palace Robot Spaceship Steamtrain Toad Wineholder Barn Caterpillar Family Ignatius
    do
        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 2 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 2 --seed 2023

        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 2 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 2 --seed 2023

        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 4 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 4 --seed 2023

        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 4 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 4 --seed 2023

        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 8 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/small.gin --scene $scene --n_features 8 --seed 2023

        echo CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 8 --seed 2023
        CUDA_VISIBLE_DEVICES=${gpu} python train.py config/base.gin --scene $scene --n_features 8 --seed 2023
    done
done
