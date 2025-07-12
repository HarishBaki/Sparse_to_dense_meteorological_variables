#!/bin/bash

#SBATCH --job-name=Barnes_interp
#SBATCH --output=slurmout/metric-%j.out
#SBATCH --error=slurmout/metric-%j.err
#SBATCH --time=02-00:00:00
# #SBATCH --mem=160gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclude=dgx02 #only needed for dgx
#SBATCH --container-name=physicsnemo
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab/Harish:/mnt/basulab/Harish,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables:/mnt/current_project,/network/rit/lab/basulab/Harish/Sparse_to_dense_meteorological_variables/Predictions:/mnt/current_project/Predictions
#SBATCH --container-workdir=/mnt/current_project
# #SBATCH --container-image=/home/harish/softwares/container_images/physicsnemo:25.03.sqsh
# #SBATCH --container-mounts=/home/harish:/home/harish,/home/harish/Ongoing_Research/Sparse_to_dense_meteorological_variables:/workspace,/data/harish/Sparse_to_dense_meteorological_variables:/workspace/data
# #SBATCH --container-workdir=/workspace

# === Activate Conda Environment ===
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate gUstNET   # Replace with your actual env name

# Call the actual training script
prediction_dir=${prediction_dir:-'Predictions'}
variable=${variable:-'i10fg'}    #'i10fg','si10','t2m','sh2'
n_inference_stations=${n_inference_stations:-'none'} # 'none' or a number
fold=${fold:-0} # Fold number for cross-validation, if applicable
mode=${mode:-'w'} # 'w' for write, 'a' for append
data_type=${data_type:-'NYSM'}

echo "----------------------------------------"
echo "Launching training with the following settings:"
echo "Prediction dir:          $prediction_dir"
echo "Variable:                $variable"
echo "Inference stations:      $n_inference_stations"
echo "Fold:                    $fold"
echo "Mode:                    $mode"
echo "Data type:               $data_type"
echo "----------------------------------------"

# Install required Python packages inside the container
pip install --quiet timm torchmetrics seaborn

# Select the script to run based on data_type
if [[ "$data_type" == "NYSM" ]]; then
    training_script="Barnes_interpolation_NYSM.py"
elif [[ "$data_type" == "RTMA" ]]; then
    training_script="Barnes_interpolation.py"
else
    echo "ERROR: Unknown data_type '$data_type'. Must be 'NYSM' or 'RTMA'."
    exit 1
fi

python $training_script \
--prediction_dir $prediction_dir \
--variable $variable \
--n_inference_stations $n_inference_stations \
--fold $fold \
--mode $mode \
