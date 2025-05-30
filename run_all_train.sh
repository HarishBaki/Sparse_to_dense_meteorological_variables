#!/bin/bash

# Set your user ID (or leave blank if you are running as your user)
USER_ID=$(whoami)

# Max number of jobs allowed in queue (8 running + 8 pending), each using 2 GPUs
MAX_JOBS=8

models=("DCNN" "UNet" "SwinT2UNet")
orography_as_channels=("false" "true")
additional_input_variabless=("si10" "si10,t2m" "si10,t2m,sh2")
train_years_ranges=("2018" "2018,2019" "2018,2020")
n_random_stationss=("50" "75" "100" "none")
weights_seeds=("43" "44" "45" "46")

for weights_seed in "${weights_seeds[@]}"; do
    for model in "${models[@]}"; do
        additional_input_variables="none"
        for orography_as_channel in "${orography_as_channels[@]}"; do
            while [ "$(squeue -u $USER_ID | grep -c ' R\| PD')" -ge "$MAX_JOBS" ]; do
                echo "Queue full (>= $MAX_JOBS jobs). Waiting..."
                sleep 60
            done
            export model orography_as_channel additional_input_variables weights_seed
            sbatch --export=All python_jobsub.slurm
        done

        orography_as_channel="true"
        for additional_input_variables in "${additional_input_variabless[@]}"; do
            # Skip condition: UNet with si10,t2m,sh2
            if [[ "$model" == "UNet" && "$additional_input_variables" == "si10,t2m,sh2" ]]; then
                echo "Skipping $model with $additional_input_variables"
                continue
            fi

            while [ "$(squeue -u $USER_ID | grep -c ' R\| PD')" -ge "$MAX_JOBS" ]; do
                echo "Queue full (>= $MAX_JOBS jobs). Waiting..."
                sleep 60
            done
            export model orography_as_channel additional_input_variables weights_seed
            sbatch --export=All python_jobsub.slurm
        done
    done
done