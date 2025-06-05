#!/bin/bash
set -e  # optional: exit on error

root_dir='/media/ssd_2tb_evo/Sparse_to_dense_meteorological_variables/WRF_simulations'

# Define simulation metadata
cases=('case_1' 'case_2' 'case_3' 'case_4')  # <-- Define this as per your actual simulation case names
start_dates=('2023-02-02_12:00:00' '2023-03-25_00:00:00' '2023-04-01_00:00:00' '2023-12-17_06:00:00')
end_dates=('2023-02-04_00:00:00' '2023-03-26_12:00:00' '2023-04-02_12:00:00' '2023-12-18_18:00:00')
cdsapirc_files=('/home/harish/.cdsapirc_harishiiitn' '/home/harish/.cdsapirc_hbakialbany' '/home/harish/.cdsapirc_hbakitudelft')

num_cdsapirc=${#cdsapirc_files[@]}

for i in "${!start_dates[@]}"; do
    case=${cases[$i]}
    echo "Running case: $case"

    # Prepare grib source directory
    grib_source_dir="$root_dir/$case/Input_DATA/ERA5"
    mkdir -p "$grib_source_dir"
    cd "$grib_source_dir"

        start_date=$(echo "${start_dates[$i]}" | sed 's/_/ /')
        end_date=$(echo "${end_dates[$i]}" | sed 's/_/ /')
        cdsapirc_file="${cdsapirc_files[$((i % num_cdsapirc))]}"

        echo "Running case: $case"
        echo "Start date: $start_date"
        echo "End date:   $end_date"
        echo "CDS API RC: $cdsapirc_file"

        # Get year, month (assume same year/month for now)
        year=$(date -ud "$start_date" +"%Y")
        month=$(date -ud "$start_date" +"%m")

        # Get list of days between start and end
        current_day=$(date -ud "$start_date" +"%Y-%m-%d")
        end_day=$(date -ud "$end_date" +"%Y-%m-%d")
        day_list=()

        while [ "$(date -ud "$current_day" +%s)" -le "$(date -ud "$end_day" +%s)" ]; do
            day_list+=("$(date -ud "$current_day" +"%d")")
            current_day=$(date -ud "$current_day +1 day" +"%Y-%m-%d")
        done

        # Join day list with commas
        days_joined=$(IFS=, ; echo "${day_list[*]}")

        echo "Year: $year, Month: $month, Days to download: $days_joined"

        # Execute the Python script
        python3 $root_dir/download_ERA5.py "$cdsapirc_file" "$year" "$month" "$days_joined" &
    cd $root_dir
done

wait
echo "All ERA5 downloads completed successfully."