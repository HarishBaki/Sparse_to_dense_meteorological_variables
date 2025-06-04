#!/bin/bash
set -e  # optional: exit on error

root_dir='/media/ssd_2tb_evo/Sparse_to_dense_meteorological_variables/WRF_simulations'

# Define simulation metadata
cases=('case_1' 'case_2' 'case_3' 'case_4')  # <-- Define this as per your actual simulation case names
start_dates=('2023-02-02_12:00:00' '2023-03-25_00:00:00' '2023-04-01_00:00:00' '2023-12-17_06:00:00')
end_dates=('2023-02-04_00:00:00' '2023-03-26_12:00:00' '2023-04-02_12:00:00' '2023-12-18_18:00:00')
cdsapirc_files=('/home/harish/.cdsapirc_harishiiitn' '/home/harish/.cdsapirc_hbakialbany' '/home/harish/.cdsapirc_hbakitudelft' '/home/harish/.cdsapirc_sandhya')

num_cdsapirc=${#cdsapirc_files[@]}

for i in "${!start_dates[@]}"; do
    case=${cases[$i]}
    echo "Running case: $case"

    # Prepare grib source directory
    grib_source_dir="$root_dir/$case/Input_DATA/ERA5"
    mkdir -p "$grib_source_dir"
    cd "$grib_source_dir"

        # Assign date ranges and .cdsapirc config
        start_date="${start_dates[$i]}"
        end_date="${end_dates[$i]}"
        cdsapirc_file="${cdsapirc_files[$((i % num_cdsapirc))]}"

        echo "Processing simulation for:"
        echo "  Start date: $start_date"
        echo "  End date:   $end_date"
        echo "  CDS API RC: $cdsapirc_file"

        # Convert underscore to space for proper date parsing
        current_date_fmt=$(echo "$start_date" | sed 's/_/ /')
        end_date_fmt=$(echo "$end_date" | sed 's/_/ /')

        while [ $(date -ud "$current_date_fmt" +%s) -le $(date -ud "$end_date_fmt" +%s) ]; do
            year=$(date -ud "$current_date_fmt" +"%Y")
            month=$(date -ud "$current_date_fmt" +"%m")
            day=$(date -ud "$current_date_fmt" +"%d")
            hour=$(date -ud "$current_date_fmt" +"%H")

            #formatted=$(date -ud "$current_date" +"%Y-%m-%d_%H:00:00")
            #echo "$formatted"
            echo $year $month $day $hour

            # Increment by 1 hour
            current_date_fmt=$(date -ud "$current_date_fmt 1 hour" +"%Y-%m-%d %H")
        done

        # Execute the Python script
        #python3 run_download_ERA5.py --root_dir "$root_dir" --start_date "$start_date" --end_date "$end_date" --cdsapirc_file "$cdsapirc_file"
    cd $root_dir
done
