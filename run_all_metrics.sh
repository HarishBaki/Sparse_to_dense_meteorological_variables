#!/bin/bash

variable='i10fg'
data_types=("RTMA")
models=("DCNN" "UNet" "SwinT2UNet")
orography_as_channels=("false" "true")
additional_input_variabless=("si10" "si10,t2m" "si10,t2m,sh2")
weights_seeds=("43" "44" "45" "46")
for weights_seed in "${weights_seeds[@]}"; do
    for data_type in "${data_types[@]}"; do
        for model in "${models[@]}"; do
            for orography_as_channel in "${orography_as_channels[@]}"; do
                additional_input_variables='none'
                export variable data_type model orography_as_channel weights_seed additional_input_variables
                echo "Running for variable $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel with weights_seed: $weights_seed and additional_input_variables: $additional_input_variables"
                sbatch --export=All jobsub_metrics.slurm
            done
            orography_as_channel='true'
            for additional_input_variables in "${additional_input_variabless[@]}"; do
                export variable data_type model orography_as_channel additional_input_variables weights_seed 
                echo "Running for variable $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables with weights_seed: $weights_seed"
                sbatch --export=All jobsub_metrics.slurm
            done
        done
    done
done

if false; then
    train_years_ranges=("2018" "2018,2019" "2018,2020")
    model="UNet"
    orography_as_channel='true'
    additional_input_variables='si10,t2m,sh2'
    for data_type in "${data_types[@]}"; do
        for train_years_range in "${train_years_ranges[@]}"; do
            export variable data_type model orography_as_channel additional_input_variables train_years_range
            export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range'
            echo "Running for variable $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range"
            sbatch --export=All jobsub_metrics.slurm
        done
    done
fi

if false; then
    model="UNet"
    train_years_range='2018,2021'
    orography_as_channel='true'
    additional_input_variables='si10,t2m,sh2'
    n_random_stationss=("none" "50" "75" "100")
    n_inference_stationss=("50" "75" "100")
    randomize_stations_persamples=("true" "false")
    inference_stations_seeds=(43 44 45 46) # Different seeds for inference stations
    for data_type in "${data_types[@]}"; do
        for inference_stations_seed in "${inference_stations_seeds[@]}"; do
            for randomize_stations_persample in "${randomize_stations_persamples[@]}"; do
                for n_random_stations in "${n_random_stationss[@]}"; do
                    for n_inference_stations in "${n_inference_stationss[@]}"; do
                        export variable data_type model orography_as_channel additional_input_variables train_years_range n_random_stations n_inference_stations randomize_stations_persample inference_stations_seed
                        sbatch --export=All jobsub_metrics.slurm
                    done
                done
            done
        done
    done
fi

if false; then  
    model="UNet"
    variable='si10'
    orography_as_channel='true'
    additional_input_variables='i10fg,t2m,sh2'
    train_years_range='2018,2021'
    n_random_stations='none'
    n_inference_stations='none'
    for data_type in "${data_types[@]}"; do
        export variable data_type model orography_as_channel additional_input_variables train_years_range n_random_stations n_inference_stations
        export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range|n_random_stations|n_inference_stations'
        echo "Running for variable: $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range, n_random_stations: $n_random_stations, n_inference_stations: $n_inference_stations"
        sbatch --export=All jobsub_metrics.slurm
    done

    model="UNet"
    variable='t2m'
    orography_as_channel='true'
    additional_input_variables='i10fg,si10,sh2'
    train_years_range='2018,2021'
    n_random_stations='none'
    n_inference_stations='none'
    for data_type in "${data_types[@]}"; do
        export variable data_type model orography_as_channel additional_input_variables train_years_range n_random_stations n_inference_stations
        export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range|n_random_stations|n_inference_stations'
        echo "Running for variable: $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range, n_random_stations: $n_random_stations, n_inference_stations: $n_inference_stations"
        sbatch --export=All jobsub_metrics.slurm
    done

    model="UNet"
    variable='sh2'
    orography_as_channel='true'
    additional_input_variables='i10fg,si10,t2m'
    train_years_range='2018,2021'
    n_random_stations='none'
    n_inference_stations='none'
    for data_type in "${data_types[@]}"; do
        export variable data_type model orography_as_channel additional_input_variables train_years_range n_random_stations n_inference_stations
        export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range|n_random_stations|n_inference_stations'
        echo "Running for variable: $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range, n_random_stations: $n_random_stations, n_inference_stations: $n_inference_stations"
        sbatch --export=All jobsub_metrics.slurm
    done
fi