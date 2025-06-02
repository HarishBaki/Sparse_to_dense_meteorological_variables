#!/bin/bash

#SBATCH --job-name=metrics_test
#SBATCH --output=slurmout/ddp_test-%j.out
#SBATCH --error=slurmout/ddp_test-%j.err
#SBATCH --time=02-00:00:00
#SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
# #SBATCH --exclude=dgx02 #only needed for dgx
# #SBATCH --container-image=/network/rit/dgx/dgx_basulab/enroot_tmp/physicsnemo:25.03.sqsh
# #SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project
# #SBATCH --container-workdir=/mnt/current_project
# #SBATCH --container-image=/home/harish/softwares/container_images/physicsnemo:25.03.sqsh
# #SBATCH --container-mounts=/home/harish:/home/harish,/home/harish/Ongoing_Research/Sparse_to_dense_meteorological_variables:/workspace,/data/harish/Sparse_to_dense_meteorological_variables:/workspace/data
# #SBATCH --container-workdir=/workspace

# === Activate Conda Environment ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gUstNET   # Replace with your actual env name

# Call the actual training script
checkpoint_dir=${checkpoint_dir:-'checkpoints'}
variable=${variable:-'i10fg'}    #'i10fg','si10','t2m','sh2'
model=${model:-'UNet'} #  'UNet' 'DCNN', 'SwinT2UNet'
orography_as_channel=${orography_as_channel:-'true'}  # 'True' or 'False'
additional_input_variables=${additional_input_variables:-'si10,t2m,sh2'} # 'si10,t2m,sh2' or 'none'
train_years_range=${train_years_range:-'2018,2021'}   # comma-seperated list of first and last years. Pass only one year if you want to train on a single year.
stations_seed=${stations_seed:-42}
n_random_stations=${n_random_stations:-'50'} # 'none' or a number
randomize_stations_persample=${randomize_stations_persample:-'false'} # 'true' or 'false'
loss=${loss:-'MaskedCharbonnierLoss'}
transform=${transform:-'standard'}  # 'standard', 'minmax', 'none'
weights_seed=${weights_seed:-42}
activation_layer=${activation_layer:-'gelu'} # 'gelu', 'relu', 'leakyrelu'
inference_stations_seed=${inference_stations_seed:-44} # Seed for inference stations, if applicable
n_inference_stations=${n_inference_stations:-'100'} # 'none' or a number
data_type=${data_type:-'RTMA'}

echo "----------------------------------------"
echo "Launching training with the following settings:"
echo "Checkpoint dir:          $checkpoint_dir"
echo "Model:                   $model"
echo "Variable:                $variable"
echo "Orography as channel:    $orography_as_channel"
echo "Additional inputs:       $additional_input_variables"
echo "Train years:             $train_years_range"
echo "Global seed:             $stations_seed"
echo "Random stations:         $n_random_stations"
echo "Randomize stations:      $randomize_stations_persample"
echo "Loss:                    $loss"
echo "Transform:               $transform"
echo "Weights seed:            $weights_seed"
echo "Activation layer:        $activation_layer"
echo "Inference stations seed: $inference_stations_seed"
echo "Inference stations:      $n_inference_stations"
echo "Data type:               $data_type"
echo "----------------------------------------"

# Install required Python packages inside the container
pip install --quiet timm torchmetrics seaborn

python compute_metrics.py \
--checkpoint_dir $checkpoint_dir \
--variable $variable \
--model $model \
--orography_as_channel $orography_as_channel \
--additional_input_variables $additional_input_variables \
--train_years_range $train_years_range \
--stations_seed $stations_seed \
--n_random_stations $n_random_stations \
--randomize_stations_persample $randomize_stations_persample \
--loss $loss \
--transform $transform \
--weights_seed $weights_seed \
--activation_layer $activation_layer \
--inference_stations_seed $inference_stations_seed \
--n_inference_stations $n_inference_stations \
--data_type $data_type 
