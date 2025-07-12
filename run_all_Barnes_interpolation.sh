#!/bin/bash

# Set your user ID (or leave blank if you are running as your user)
USER_ID=$(whoami)

# Max number of jobs allowed in queue (8 running + 8 pending), each using 2 GPUs
MAX_JOBS=8

# === Variable names (external and internal) ===
if false; then
    stations_seeds=(43 44 45 46)
    stations=(50 75 100)
    var_names=("i10fg")

    for stations_seed in "${stations_seeds[@]}"; do
        for var_name in "${var_names[@]}"; do
            # Set mode: 'w' for i10fg (first write), 'a' for all others
            if [[ "$var_name" == "i10fg" ]]; then
                mode="w"
            else
                mode="a"
            fi
            for n_staitons in "${stations[@]}"; do
                echo "Processing variable: $var_name, stations_seed $stations_seed, n_stations $n_staitons, with mode: $mode"
                # Example: run test script
                #python Barnes_interpolation.py $var_name $stations_seed $n_staitons $mode
                python Barnes_interpolation_NYSM.py $var_name $stations_seed $n_staitons $mode  
            done
            echo "Variable $var_name processed."
        done
    done
fi

n_inference_stationss=("none" "93" "62")
data_types=("NYSM")
var_names=("i10fg")
for var_name in "${var_names[@]}"; do
    # Set mode: 'w' for i10fg (first write), 'a' for all others
    if [[ "$var_name" == "i10fg" ]]; then
        mode="w"
    else
        mode="a"
    fi
    for data_type in "${data_types[@]}"; do
        for n_inference_stations in "${n_inference_stationss[@]}"; do
            if [[ "$n_inference_stations" == "93" ]]; then
                folds=(0 1 2 3)
            elif [[ "$n_inference_stations" == "62" ]]; then
                folds=(0 1 2 3 4 5)
            else
                folds=(0)  # For 'none', we don't need to loop through folds
            fi
            for fold in "${folds[@]}"; do
                echo "Running inference with n_inference_stations=$n_inference_stations, data_type=$data_type, fold=$fold, var_name=$var_name, mode=$mode"
                export var_name mode data_type n_inference_stations fold
                while [ "$(squeue -u $USER_ID | grep -c ' R\| PD')" -ge "$MAX_JOBS" ]; do
                    echo "Queue full (>= $MAX_JOBS jobs). Waiting..."
                    sleep 60
                done
                sbatch --export=All jobsub_Barnes_interpolation.slurm
            done
        done
    done
done
