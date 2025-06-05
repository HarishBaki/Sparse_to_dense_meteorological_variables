WPS_source_dir=/media/sukanta/HD2/WRF/WRFV4.4/WPS # this is the WPS installation directory

root_dir='/media/ssd_2tb_evo/Sparse_to_dense_meteorological_variables/WRF_simulations'

# Define simulation metadata
cases=('case_1' 'case_2' 'case_3' 'case_4')  # <-- Define this as per your actual simulation case names
start_dates=('2023-02-02_12:00:00' '2023-03-25_00:00:00' '2023-04-01_00:00:00' '2023-12-17_06:00:00')
end_dates=('2023-02-04_00:00:00' '2023-03-26_12:00:00' '2023-04-02_12:00:00' '2023-12-18_18:00:00')

for (( i=0; i<${#cases[@]}; i++ )); # remember, 0 means FLLJ_1
do
    case=${cases[$i]}
    echo $case
    # there are folders in the WPS directory, which are ends with _geogrid. read them
    runs=('run_1')     # actual runs ('run_1' 'run_2,3,4' 'run_5' 'run_6' 'run_7' 'run_8')
    # loop over the geogrid_dirs
    for run in ${runs[@]};
    do
        geogrid_dir=$root_dir/$run"_geogrid"
        # from the geogrid_dir, exclude the _geogrid part, and extract the WPS directory name
        # only read the last part of the path, which is the directory name
        WPS_dir="WPS_"$run
        # create the WPS run directory
        run_dir=$root_dir/$case/$WPS_dir
        mkdir -p $run_dir
        cd $run_dir
            # Call the WPS pipeline here
            grib_prefix='ERA5'
            grib_source_dir=$root_dir/$case/Input_DATA/ERA5
            Vtable='Vtable.ERA-interim.pl'
            bash $root_dir/WPS_pipeline.sh $root_dir $WPS_source_dir $geogrid_dir ${start_dates[$i]} ${end_dates[$i]} $grib_source_dir $grib_prefix $Vtable &
            echo $root_dir $WPS_source_dir $geogrid_dir ${start_dates[$i]} ${end_dates[$i]} $grib_source_dir $grib_prefix $Vtable
        cd $root_dir
    done
    wait
done
wait
echo "WPS pipeline completed" 
