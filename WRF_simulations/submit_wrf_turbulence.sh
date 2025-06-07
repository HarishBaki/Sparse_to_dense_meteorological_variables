#!/bin/bash
#SBATCH --job-name=wrf
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000

date

# Use SLURM_NTASKS to automatically match the --ntasks value
mpirun -np $SLURM_NTASKS ./real.exe
mpirun -np $SLURM_NTASKS ./wrf.exe

date