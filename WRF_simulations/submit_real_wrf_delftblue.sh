#!/bin/sh
#
#SBATCH --job-name="case_name"
#SBATCH --partition=compute
#SBATCH --time=120:00:00
#SBATCH --ntasks=96
#sbatch --cpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-ceg-grs

module load 2023r1
module load openmpi/4.1.4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun ./real.exe	#Executing real.exe
if ls rsl.out.0000 1>/dev/null 2>&1; #Checking the existance of rsl.out.0000 file
then
	# Checking if the file ends with "real_em: SUCCESS COMPLETE REAL_EM INIT"
	if tail -n 1 rsl.out.0000 | grep -q "real_em: SUCCESS COMPLETE REAL_EM INIT"; then
		#rm met_em.d0*	#Removing the metfiles
		srun ./wrf.exe	#Executing wrf.exe
		if tail -n 1 rsl.out.0000 | grep -q "wrf: SUCCESS COMPLETE WRF"; then
			echo "WRF finished" > wrf.out
		else
			echo "WRF unfinished" > wrf.out
		fi
	else
		echo "REAL unfinished" > real.out	
	fi
else
	echo "rsl.out.0000 doesn't exist"	
fi

wait
