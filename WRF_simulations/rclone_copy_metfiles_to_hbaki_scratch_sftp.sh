root_dir=$(pwd)

# Define simulation metadata
cases=('case_1' 'case_2' 'case_3' 'case_4')  # <-- Define this as per your actual simulation case names
start_dates=('2023-02-02_12:00:00' '2023-03-25_00:00:00' '2023-04-01_00:00:00' '2023-12-17_06:00:00')
end_dates=('2023-02-04_00:00:00' '2023-03-26_12:00:00' '2023-04-02_12:00:00' '2023-12-18_18:00:00')

#for (( i=0; i<${#cases[@]}; i++ )); do
i=0
    case=${cases[$i]}
    echo $case
    # there are folders in the WPS directory, which are ends with _geogrid. read them
    runs=('run_1')     # actual runs ('run_1' 'run_2,3,4' 'run_5' 'run_6' 'run_7' 'run_8')
    # loop over the geogrid_dirs
    for run in ${runs[@]};
    do
		WPS_dir="WPS_"$run
        # create the WPS run directory
        run_dir=$root_dir/$case/$WPS_dir
		#rclone copy --progress --transfers 12 $run_dir/metfiles hbaki_scratch_sftp:/scratch/hbaki/Sparse_to_dense_meteorological_variables/WRF_simulations/$case/$WPS_dir/metfiles/
    rclone copy --progress --transfers 12 $run_dir/metfiles hb533188_dgxhead:/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables/WRF_simulations/$case/$WPS_dir/metfiles/
    done
#done
echo "WPS pipeline completed" 