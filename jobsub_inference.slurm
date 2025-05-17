#!/bin/bash

#SBATCH --job-name=ddp_test
#SBATCH --output=slurmout/ddp_test-%j.out
#SBATCH --error=slurmout/ddp_test-%j.err
#SBATCH --time=02-00:00:00
#SBATCH --mem=40gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
# #SBATCH --exclude=dgx02 #only needed for dgx
# #SBATCH --container-image=/network/rit/dgx/dgx_basulab/enroot_tmp/physicsnemo:25.03.sqsh
# #SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
# #SBATCH --container-workdir=/mnt/current_project
#SBATCH --container-image=/home/harish/softwares/container_images/physicsnemo:25.03.sqsh
#SBATCH --container-mounts=/home/harish:/home/harish,/home/harish/Ongoing_Research/Sparse_to_dense_meteorological_variables:/workspace,/data/harish/Sparse_to_dense_meteorological_variables:/workspace/data
#SBATCH --container-workdir=/workspace

# Optionally pass SLURM values into the shell script as environment variables
export nproc_per_node=${SLURM_NTASKS_PER_NODE}
export num_workers=${SLURM_CPUS_PER_TASK}
export MASTER_PORT=$((20000 + RANDOM % 20000))

# Call the actual training script
nproc_per_node=${nproc_per_node:-1}
num_workers=${num_workers:-32}
checkpoint_dir=${checkpoint_dir:-'checkpoints'}
variable=${variable:-'i10fg'}    #'i10fg','si10','t2m','sh2'
model=${model:-'UNet'} #  'UNet' 'DCNN', 'SwinT2UNet'
orography_as_channel=${orography_as_channel:-'true'}  # 'True' or 'False'
additional_input_variables=${additional_input_variables:-'none'} # 'si10,t2m,sh2' or 'none'
train_years_range=${train_years_range:-'2018,2021'}   # comma-seperated list of first and last years. Pass only one year if you want to train on a single year.
stations_seed=${stations_seed:-42}
n_random_stations=${n_random_stations:-'none'} # 'none' or a number
loss=${loss:-'MaskedCharbonnierLoss'}
transform=${transform:-'standard'}  # 'standard', 'minmax', 'none'
epochs=${epochs:-120}
batch_size=${batch_size:-24}
wandb_id=${wandb_id:-'none'}
weights_seed=${weights_seed:-42}
activation_layer=${activation_layer:-'gelu'} # 'gelu', 'relu', 'leakyrelu'
n_inference_stations=${n_inference_stations:-'none'} # 'none' or a number

echo "----------------------------------------"
echo "Launching training with the following settings:"
echo "Model:                   $model"
echo "Variable:                $variable"
echo "Orography as channel:    $orography_as_channel"
echo "Additional inputs:       $additional_input_variables"
echo "Train years:             $train_years_range"
echo "Global seed:             $stations_seed"
echo "Random stations:         $n_random_stations"
echo "Loss:                    $loss"
echo "Transform:               $transform"
echo "Epochs:                  $epochs"
echo "Batch size:              $batch_size"
echo "Num workers:             $num_workers"
echo "Checkpoint dir:          $checkpoint_dir"
echo "WandB ID:                $wandb_id"
echo "Weights seed:            $weights_seed"
echo "Activation layer:        $activation_layer"
echo "Inference stations:      $n_inference_stations"
echo "----------------------------------------"
echo "NProc per node:          $nproc_per_node"
echo "Master port:             $MASTER_PORT"
echo "----------------------------------------"

# Install required Python packages inside the container
pip install --quiet timm torchmetrics seaborn

torchrun --nproc_per_node=$nproc_per_node --master_port=$MASTER_PORT RTMA_inference.py \
--checkpoint_dir $checkpoint_dir \
--variable $variable \
--model $model \
--orography_as_channel $orography_as_channel \
--additional_input_variables $additional_input_variables \
--train_years_range $train_years_range \
--stations_seed $stations_seed \
--n_random_stations $n_random_stations \
--loss $loss \
--transform $transform \
--epochs $epochs \
--batch_size $batch_size \
--num_workers $num_workers \
--wandb_id $wandb_id \
--weights_seed $weights_seed \
--activation_layer $activation_layer \
--n_inference_stations $n_inference_stations
#--resume
