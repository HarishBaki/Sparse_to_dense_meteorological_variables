#!/bin/bash

#SBATCH --job-name=ddp_test
#SBATCH --output=slurmout/ddp_test-%j.out
#SBATCH --error=slurmout/ddp_test-%j.err
#SBATCH --time=02-00:00:00
#SBATCH --mem=40gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=36
#SBATCH --container-image=/home/harish/softwares/container_images/physicsnemo:25.03.sqsh
#SBATCH --container-mounts=/home/harish:/home/harish,/home/harish/Ongoing_Research/Sparse_to_dense_meteorological_variables:/workspace,/data/harish/Sparse_to_dense_meteorological_variables:/workspace/data,/data/harish/Gust_field_nowcasting_from_Sparse_stations:/data/harish/Gust_field_nowcasting_from_Sparse_stations
#SBATCH --container-workdir=/workspace

# Install required Python packages inside the container
pip install --quiet timm torchmetrics seaborn
  
year=2023
python NYSM_5min_frequency_inference_from_best_model.py --year "$year"
