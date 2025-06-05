#!/bin/bash
#SBATCH --job-name=wrf
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4000
date
mpirun -np 8 ./real.exe
mpirun -np 8 ./wrf.exe
date
