#!/bin/bash
# DCNN and all variables are not submitted. 
variables=("si10" "t2m" "sh2")
models=("DCNN" "UNet" "SwinT2UNet")
orography_as_channels=("false" "true")
additional_input_variabless=("si10" "si10,t2m" "si10,t2m,sh2")
years=("2018" "2018,2019" "2018,2020")
n_stations=("50" "75" "100")
weights_seeds=("43" "44" "45" "46")

model='DCNN'
#for model in "${models[@]}"; do
for orography_as_channel in "${orography_as_channels[@]}"; do
    export model orography_as_channel
    sbatch --export=All python_jobsub.slurm   
    #bash python_jobsub.slurm
done

orography_as_channel='true'
for additional_input_variables in "${additional_input_variabless[@]}"; do
    export model orography_as_channel additional_input_variables
    sbatch --export=All python_jobsub.slurm   
    #bash python_jobsub.slurm
done