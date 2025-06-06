#!/bin/bash
if false; then 
    n_random_stationss=("50" "75" "100" "none")
    randomize_stations_persamples=("true" "false")
    n_inference_stationss=("50" "75" "100")
    orography_as_channel='true'
    additional_input_variables='si10,t2m,sh2'
    train_years_range='2018,2021'
    inference_stations_seeds=(43 44 45 46) # Different seeds for inference stations
    for inference_stations_seed in "${inference_stations_seeds[@]}"; do
        for randomize_stations_persample in "${randomize_stations_persamples[@]}"; do
            for n_random_stations in "${n_random_stationss[@]}"; do
                for n_inference_stations in "${n_inference_stationss[@]}"; do
                    echo "Running inference with randomize_stations_persample=$randomize_stations_persample, n_random_stations=$n_random_stations, n_inference_stations=$n_inference_stations, inference_stations_seed=$inference_stations_seed"
                    export orography_as_channel additional_input_variables train_years_range n_random_stations randomize_stations_persample n_inference_stations inference_stations_seed
                    sbatch --export=All jobsub_inference.slurm
                done
            done
        done
    done
fi

if false; then
    models=("DCNN" "UNet" "SwinT2UNet")
    orography_as_channels=("false" "true")
    additional_input_variabless=("si10" "si10,t2m" "si10,t2m,sh2")
    weights_seeds=("43" "44" "45" "46")
    train_years_range='2018,2021'
    for weights_seed in "${weights_seeds[@]}"; do
        for model in "${models[@]}"; do
            additional_input_variables="none"
            for orography_as_channel in "${orography_as_channels[@]}"; do
                export model orography_as_channel additional_input_variables weights_seed
                sbatch --export=All jobsub_inference.slurm
            done

            orography_as_channel="true"
            for additional_input_variables in "${additional_input_variabless[@]}"; do
                # Skip condition: UNet with si10,t2m,sh2
                if [[ "$model" == "UNet" && "$additional_input_variables" == "si10,t2m,sh2" ]]; then
                    echo "Skipping $model with $additional_input_variables"
                    continue
                fi
                export model orography_as_channel additional_input_variables weights_seed
                sbatch --export=All jobsub_inference.slurm
            done
        done
    done
fi

losses=("MaskedMSELoss" "MaskedCombinedMAEQuantileLoss")
n_inference_stationss=("50" "75" "100" "none")
for loss in "${losses[@]}"; do
    for n_inference_stations in "${n_inference_stationss[@]}"; do
        n_random_stations='none'  # Set to 'none' for inference
        randomize_stations_persample='false'  # Set to 'true' for inference
        export n_random_stations randomize_stations_persample n_inference_stations loss
        sbatch --export=All jobsub_inference.slurm

        n_random_stations='none'
        randomize_stations_persample='true'  # Set to 'true' for inference
        export n_random_stations randomize_stations_persample n_inference_stations loss
        sbatch --export=All jobsub_inference.slurm

        n_random_stations='75'  # 
        randomize_stations_persample='true'  # Set to 'true' for inference
        export n_random_stations randomize_stations_persample n_inference_stations loss
        sbatch --export=All jobsub_inference.slurm
    done
done