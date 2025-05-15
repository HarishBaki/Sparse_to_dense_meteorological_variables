#!/bin/bash
# DCNN and all variables are not submitted. 
variables=("si10" "t2m" "sh2")
models=("DCNN" "UNet" "SwinT2UNet")
orography_as_channels=("false" "true")
additional_input_variabless=("si10" "si10,t2m" "si10,t2m,sh2")
train_years_ranges=("2018" "2018,2019" "2018,2020")
n_random_stationss=("50" "75" "100")
weights_seeds=("42" "43" "44" "45" "46")

for model in "${models[@]}"; do
    train_years_range="2018,2021"
    n_random_stations='none'
    for orography_as_channel in "${orography_as_channels[@]}"; do
        export model orography_as_channel train_years_range n_random_stations
        #sbatch --export=All python_jobsub.slurm
        bash python_jobsub.slurm
    done
    orography_as_channel='true'
    for additional_input_variables in "${additional_input_variabless[@]}"; do
        export model orography_as_channel additional_input_variables train_years_range n_random_stations
        #sbatch --export=All python_jobsub.slurm
        bash python_jobsub.slurm
    done
done
model="UNet"
orography_as_channel='true'
additional_input_variables='si10,t2m,sh2'
for train_years_range in "${train_years_ranges[@]}"; do
    export model orography_as_channel additional_input_variables train_years_range n_random_stations
    #sbatch --export=All python_jobsub.slurm
    bash python_jobsub.slurm
done
train_years_range='2018,2021'
for n_random_stations in "${n_random_stationss[@]}"; do
    export model orography_as_channel additional_input_variables train_years_range n_random_stations
    #sbatch --export=All python_jobsub.slurm
    bash python_jobsub.slurm
done
  