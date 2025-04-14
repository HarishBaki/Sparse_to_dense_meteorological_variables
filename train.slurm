#!/bin/bash

#SBATCH --job-name=ddp_train
#SBATCH --output=slurmout/ddp_train-%j.out
#SBATCH --error=slurmout/ddp_train-%j.err
#SBATCH --time=00-08:00:00
#SBATCH --mem=640gb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --container-image='docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03'
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
#SBATCH --container-workdir=/mnt/current_project

# === DDP launch ===
MASTER_ADDR=$(hostname -s)
NODE_RANK=$SLURM_NODEID

torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  train.py --variable i10fg --epochs 20 --resume