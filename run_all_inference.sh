#!/bin/bash
n_random_stationss=("50" "75" "100" "none")
n_inference_stationss=("50" "75" "100" "none")
orography_as_channel='true'
additional_input_variables='si10,t2m,sh2'
train_years_range='2018,2021'
for n_random_stations in "${n_random_stationss[@]}"; do
    for n_inference_stations in "${n_inference_stationss[@]}"; do
        export orography_as_channel additional_input_variables train_years_range n_random_stations n_inference_stations
        sbatch --export=All jobsub_inference.slurm
    done
done
  