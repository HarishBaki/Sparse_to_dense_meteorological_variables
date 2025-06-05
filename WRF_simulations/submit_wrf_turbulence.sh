#!/bin/bash
#SBATCH --job-name=wrf
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=4000

date

# Use SLURM_NTASKS to automatically match the --ntasks value
mpirun -np $SLURM_NTASKS ./real.exe
mpirun -np $SLURM_NTASKS ./wrf.exe

date