#!/bin/bash
n_random_stationss=("50" "75" "100" "none")
randomize_stations_persamples=("true" "false")
n_inference_stationss=("50" "75" "100" "none")
orography_as_channel='true'
additional_input_variables='si10,t2m,sh2'
train_years_range='2018,2021'
for randomize_stations_persample in "${randomize_stations_persamples[@]}"; do
    for n_random_stations in "${n_random_stationss[@]}"; do
        for n_inference_stations in "${n_inference_stationss[@]}"; do
            echo "Running inference with randomize_stations_persample=$randomize_stations_persample, n_random_stations=$n_random_stations, n_inference_stations=$n_inference_stations"
            export orography_as_channel additional_input_variables train_years_range n_random_stations randomize_stations_persample n_inference_stations
            sbatch --export=All jobsub_inference.slurm
        done
    done
done