#!/bin/bash

#SBATCH --job-name=ddp_train
#SBATCH --output=slurmout/ddp_train-%j.out
#SBATCH --error=slurmout/ddp_train-%j.err
#SBATCH --time=01-00:00:00
#SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --container-image='docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03'
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
#SBATCH --container-workdir=/mnt/current_project

torchrun \
    --master_port=$((12000 + RANDOM % 1000)) \
    --nproc_per_node=2 \
    train.py --variable sh2 --epochs 120