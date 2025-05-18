#!/bin/bash
variable='i10fg'
data_types=("RTMA" "NYSM")
models=("DCNN" "UNet" "SwinT2UNet")
orography_as_channels=("false" "true")
additional_input_variabless=("si10" "si10,t2m" "si10,t2m,sh2")

for data_type in "${data_types[@]}"; do
    for model in "${models[@]}"; do
        for orography_as_channel in "${orography_as_channels[@]}"; do
            export variable data_type model orography_as_channel
            export -p | grep -E 'variable|data_type|model|orography_as_channel'
            echo "Running for variable $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel"
            sbatch --export=All jobsub_metrics.slurm
        done
        orography_as_channel='true'
        for additional_input_variables in "${additional_input_variabless[@]}"; do
            export variable data_type model orography_as_channel additional_input_variables
            export -p | grep -E 'variable|variable|data_type|model|orography_as_channel|additional_input_variables'
            echo "Running for variable $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables"
            sbatch --export=All jobsub_metrics.slurm
        done
    done
done

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

model="UNet"
train_years_range='2018,2021'
orography_as_channel='true'
additional_input_variables='si10,t2m,sh2'
n_random_stationss=("none" "50" "75" "100")
n_inference_stationss=("none" "50" "75" "100")
for data_type in "${data_types[@]}"; do
    for n_random_stations in "${n_random_stationss[@]}"; do
        for n_inference_stations in "${n_inference_stationss[@]}"; do
            export variable data_type model orography_as_channel additional_input_variables train_years_range n_random_stations n_inference_stations
            export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range|n_random_stations|n_inference_stations'
            echo "Running for variable $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range, n_random_stations: $n_random_stations, n_inference_stations: $n_inference_stations"
            sbatch --export=All jobsub_metrics.slurm
        done
    done
done
  
model="UNet"
variable='si10'
orography_as_channel='true'
additional_input_variables='i10fg,t2m,sh2'
train_years_range='2018,2021'
for data_type in "${data_types[@]}"; do
    export variable data_type model orography_as_channel additional_input_variables train_years_range
    export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range'
    echo "Running for variable: $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range"
    sbatch --export=All jobsub_metrics.slurm
done

model="UNet"
variable='t2m'
orography_as_channel='true'
additional_input_variables='i10fg,si10,sh2'
train_years_range='2018,2021'
for data_type in "${data_types[@]}"; do
    export variable data_type model orography_as_channel additional_input_variables train_years_range
    export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range'
    echo "Running for variable: $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range"
    sbatch --export=All jobsub_metrics.slurm
done

model="UNet"
variable='sh2'
orography_as_channel='true'
additional_input_variables='i10fg,si10,t2m'
train_years_range='2018,2021'
for data_type in "${data_types[@]}"; do
    export variable data_type model orography_as_channel additional_input_variables train_years_range
    export -p | grep -E 'variable|data_type|model|orography_as_channel|additional_input_variables|train_years_range'
    echo "Running for variable: $variable, data_type: $data_type, model: $model, orography_as_channel: $orography_as_channel, additional_input_variables: $additional_input_variables, train_years_range: $train_years_range"
    sbatch --export=All jobsub_metrics.slurm
done
