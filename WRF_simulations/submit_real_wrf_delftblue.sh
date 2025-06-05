#!/bin/sh
#
#SBATCH --job-name="case_name"
#SBATCH --partition=compute
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-ceg-grs

module load 2022r2
module load openmpi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

Udrive_dir="$1"
case="$2"
run_dir="$3"

srun ./real.exe	#Executing real.exe
if ls rsl.out.0000 1>/dev/null 2>&1; #Checking the existance of rsl.out.0000 file
then
	# Checking if the file ends with "real_em: SUCCESS COMPLETE REAL_EM INIT"
	if tail -n 1 rsl.out.0000 | grep -q "real_em: SUCCESS COMPLETE REAL_EM INIT"; then
		#rm met_em.d0*	#Removing the metfiles
		srun ./wrf.exe	#Executing wrf.exe
		if tail -n 1 rsl.out.0000 | grep -q "wrf: SUCCESS COMPLETE WRF"; then
			bash rclone_copy_wrfout_to_HBaki_sftp.sh "$Udrive_dir" "$case" "$run_dir"
		else
			echo "WRF $case $run_dir unfinished" > wrf.out
		fi
	else
		echo "REAL $case $run_dir unfinished" > real.out	
	fi
else
	echo "rsl.out.0000 doesn't exist"	
fi

wait
