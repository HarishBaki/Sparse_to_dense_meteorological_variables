#!/bin/bash

#SBATCH --job-name=ddp_train
#SBATCH --output=slurmout/ddp_train-%j.out
#SBATCH --error=slurmout/ddp_train-%j.err
#SBATCH --time=02-00:00:00
#SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --container-image='docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03'
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
#SBATCH --container-workdir=/mnt/current_project

# ===== Variables =====
nproc_per_node=${SLURM_NTASKS_PER_NODE}
variable='i10fg'
epochs=120
checkpoint_dir='checkpoints'
loss='MaskedCharbonnierLoss'
model='DCNN'   # or 'UNet', 'DCNN', 'GoogleUNet', 'SwinT2UNet'
batch_size=16
transform='standard'
train_years_range='2018,2021'
# Uncomment to resume from checkpoint
#resume='--resume'

# ===== Launch Training =====
torchrun \
    --master_port=$((12000 + RANDOM % 1000)) \
    --nproc_per_node=$nproc_per_node \
    train.py \
    --variable $variable \
    --epochs $epochs \
    --checkpoint_dir $checkpoint_dir \
    --batch_size $batch_size \
    --model $model \
    --loss $loss \
    --transform $transform \
    --train_years_range $train_years_range
    #$resume