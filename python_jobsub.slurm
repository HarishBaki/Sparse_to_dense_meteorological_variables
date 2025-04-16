#!/bin/bash

#SBATCH --job-name=ddp_train
#SBATCH --output=slurmout/ddp_train-%j.out
#SBATCH --error=slurmout/ddp_train-%j.err
#SBATCH --time=00-01:00:00
#SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --container-image='docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03'
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
#SBATCH --container-workdir=/mnt/current_project

# === DDP launch ===
MASTER_ADDR=$(hostname -s)
NODE_RANK=$SLURM_NODEID

python data_loader.py