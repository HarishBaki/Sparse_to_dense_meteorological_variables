#!/bin/bash

#SBATCH --job-name=ddp_train
#SBATCH --output=slurmout/ddp_train-%j.out
#SBATCH --error=slurmout/ddp_train-%j.err
#SBATCH --time=02-00:00:00
#SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=dgx02
# #SBATCH --container-image='docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03'
#SBATCH --container-image='/network/rit/dgx/dgx_basulab/enroot_tmp/physicsnemo:25.03.sqsh'
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
#SBATCH --container-workdir=/mnt/current_project

# Optionally pass SLURM values into the shell script as environment variables
export nproc_per_node=${SLURM_NTASKS_PER_NODE}
export num_workers=${SLURM_CPUS_PER_TASK}
export MASTER_PORT=$((20000 + RANDOM % 20000))

# Call the actual training script
nproc_per_node=${nproc_per_node:-2}
num_workers=${num_workers:-32}
variable='i10fg'    #'i10fg','d2m','t2m','si10','sh2','sp'
epochs=120
checkpoint_dir='checkpoints'
loss='MaskedCharbonnierLoss'
model='DCNN' #  'UNet' 'DCNN', 'GoogleUNet', 'SwinT2UNet'
batch_size=16
transform='standard'  # 'standard', 'minmax', 'none'
train_years_range='2018,2021'   # comma-seperated list of first and last years. Pass only one year if you want to train on a single year.
wandb_id='jtqxcbcw'
#additional_input_variables='si10' #'si10,t2m,d2m,sh2'
global_seed=42
n_random_stations=100

torchrun --nproc_per_node=$nproc_per_node --master_port=$MASTER_PORT train.py \
--variable $variable \
--epochs $epochs \
--checkpoint_dir $checkpoint_dir \
--batch_size $batch_size \
--model $model \
--loss $loss \
--transform $transform \
--train_years_range $train_years_range \
--num_workers $num_workers \
--global_seed $global_seed \
--n_random_stations $n_random_stations
#--wandb_id $wandb_id 
#--additional_input_variables $additional_input_variables 
#--resume