#!/bin/bash
# DCNN and all variables are not submitted. 
variables=("si10" "t2m" "sh2")
models=("DCNN" "UNet" "SwinT2UNet")
orography_as_channels=("false" "true")
additional_input_variabless=("si10" "si10,t2m" "si10,t2m,sh2")
train_years_ranges=("2018" "2018,2019" "2018,2020")
n_random_stationss=("50" "75" "100")
weights_seeds=("43" "44" "45" "46")

model='UNet'
orography_as_channel='true'
additional_input_variables='si10,t2m,sh2'
for n_random_stations in "${n_random_stationss[@]}"; do
    export model orography_as_channel additional_input_variables n_random_stations
    sbatch --export=All python_jobsub.slurm   
    #bash python_jobsub.slurm
done