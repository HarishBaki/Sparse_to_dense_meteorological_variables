root_dir=$(pwd)

# Define simulation metadata
cases=('case_1' 'case_2' 'case_3' 'case_4')  # <-- Define this as per your actual simulation case names
start_dates=('2023-02-02_18:00:00' '2023-03-25_06:00:00' '2023-04-01_06:00:00' '2023-12-17_12:00:00')
end_dates=('2023-02-04_00:00:00' '2023-03-26_12:00:00' '2023-04-02_12:00:00' '2023-12-18_18:00:00')
#for ((i=1;i<${#cases[@]};++i)); do
i=1
	case=${cases[$i]}
	start_date=$(echo "${start_dates[$i]}" | sed 's/_/ /')
	end_date=$(echo "${end_dates[$i]}" | sed 's/_/ /')

	start_year=$(date -ud "$start_date" +"%Y")
	start_month=$(date -ud "$start_date" +"%m")
	start_day=$(date -ud "$start_date" +"%d")
	start_hour=$(date -ud "$start_date" +"%H")
	end_year=$(date -ud "$end_date" +"%Y")
	end_month=$(date -ud "$end_date" +"%m")
	end_day=$(date -ud "$end_date" +"%d")
	end_hour=$(date -ud "$end_date" +"%H")

	# call WRF pipeline here
	for run in 2 4 5; do
		run_dir=$case/WRF_run_$run

		mkdir -p $run_dir
		cd $run_dir
			metgrid_dir=$root_dir/$case/WPS_run_1/metfiles
			# link metfiles
			ln -sf $metgrid_dir/* .
			
			#cp -r $WRF_DIR/run/* .
			rsync -av --exclude='namelist.input' --exclude='*.exe' $WRF_DIR/run/* .
			#rm *.exe
			#rm namelist.input
			ln -sf $WRF_DIR/main/*.exe .
			
			cp $root_dir"/namelists/run_$run"* namelist.input
			sed -i -e "s/ start_year = / start_year = $start_year,$start_year,$start_year,/g" namelist.input
			sed -i -e "s/ start_month = / start_month = $start_month,$start_month,$start_month,/g" namelist.input
			sed -i -e "s/ start_day = / start_day = $start_day,$start_day,$start_day,/g" namelist.input
			sed -i -e "s/ start_hour = / start_hour = $start_hour,$start_hour,$start_hour,/g" namelist.input
			sed -i -e "s/ end_year = / end_year = $end_year,$end_year,$end_year,/g" namelist.input
			sed -i -e "s/ end_month = / end_month = $end_month,$end_month,$end_month,/g" namelist.input
			sed -i -e "s/ end_day = / end_day = $end_day,$end_day,$end_day,/g" namelist.input
			sed -i -e "s/ end_hour = / end_hour = $end_hour,$end_hour,$end_hour,/g" namelist.input	
			
			cp $root_dir/tslist .
			cp $root_dir/myoutfields.txt .
			
			cp $root_dir/submit_wrf_turbulence.sh .
			#cp $root_dir/submit_real_wrf_delftblue.sh .
			sbatch submit_wrf_turbulence.sh
			#sbatch submit_real_wrf_delftblue.sh
		cd $root_dir
	done
#done
