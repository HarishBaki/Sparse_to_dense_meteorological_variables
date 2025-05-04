#!/bin/bash
# DCNN and all variables are not submitted. 
variables=("si10" "t2m" "sh2")
models=("DCNN" "UNet" "SwinT2UNet")
orographies=("false" "true")
add_vars=("si10" "si10,t2m" "si10,t2m,sh2")
years=("2018" "2018,2019" "2018,2020")
n_stations=("50" "75" "100")

for model in "${models[@]}"; do
  for variable in "${variables[@]}"; do
      export variable model
      sbatch --export=All python_jobsub.slurm   # bash python_jobsub.slurm for bash script.
  done
done