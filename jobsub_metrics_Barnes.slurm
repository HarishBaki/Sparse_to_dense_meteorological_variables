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
variable=${variable:-'i10fg'}    #'i10fg','si10','t2m','sh2'
inference_stations_seed=${inference_stations_seed:-42} # Seed for inference stations, if applicable
n_inference_stations=${n_inference_stations:-'none'} # 'none' or a number
data_type=${data_type:-'RTMA'}

echo "----------------------------------------"
echo "Launching training with the following settings:"
echo "Variable:                $variable"
echo "Inference stations seed: $inference_stations_seed"
echo "Inference stations:      $n_inference_stations"
echo "Data type:               $data_type"
echo "----------------------------------------"

# Install required Python packages inside the container
pip install --quiet timm torchmetrics seaborn

python compute_metrics_Barnes.py \
--variable $variable \
--inference_stations_seed $inference_stations_seed \
--n_inference_stations $n_inference_stations \
--data_type $data_type 
