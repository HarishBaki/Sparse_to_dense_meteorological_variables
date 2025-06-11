#!/bin/bash
#SBATCH --job-name=real-wrf
#SBATCH --output=slurm-real-wrf-%j.out
#SBATCH --error=slurm-real-wrf-%j.err
#SBATCH --partition=burst-basu
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=60:00

ulimit -s unlimited
ulimit -l unlimited

export WRFIO_NCD_LARGE_FILE_SUPPORT=1

module load intel-2022/compiler
module load intel-2022/icc
#export PATH=/network/rit/lab/basulab/OpenMPI-4.1.5/bin:$PATH
#export LD_LIBRARY_PATH=/network/rit/lab/basulab/OpenMPI-4.1.5/lib:$LD_LIBRARY_PATH

time mpirun ./real.exe
time mpirun ./wrf.exe
echo "WRF simulation completed successfully."
