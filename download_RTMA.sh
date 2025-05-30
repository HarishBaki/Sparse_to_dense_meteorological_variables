#!/bin/bash

# ===== User-configurable concurrent jobs =====
MAX_JOBS=96  # Set this to desired number of parallel Python jobs

# Variables to be passed to Python script
VARIABLES=("GUST" "HGT" "TMP" "WIND" "WDIR" "DPT" "SPFH" "PRES") 
LEVELS=("10 m" "surface" "2 m" "10 m" "10 m" "2 m" "2 m" "surface")
# Loop over each index in the VARIABLES array
for i in "${!VARIABLES[@]}"; do
	VARIABLE="${VARIABLES[i]}"
	LEVEL="${LEVELS[i]}"
	DOWNLOAD_PATH="/data/harish/RTMA/$VARIABLE"

	# Loop over the years 2018 to 2023
	for year in {2024..2024}; do
	  # Loop over the months 1 to 12
	  for month in $(seq -w 1 12); do
	    # Determine the number of days in the month, accounting for leap years
	    if [[ "$month" == "04" || "$month" == "06" || "$month" == "09" || "$month" == "11" ]]; then
	    	days_in_month=30
	    elif [[ "$month" == "02" ]]; then
	      # Check for leap year (divisible by 4 and not 100 unless also divisible by 400)
	    	if (( (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) )); then
				days_in_month=29
	    	else
				days_in_month=28
	    	fi
	    else
	    	days_in_month=31
	    fi
	    # Loop over the days of the month
	    for day in $(seq -w 1 $days_in_month); do
	      # Loop over the hours of the day
	      for hour in $(seq -w 0 23); do
			# Call the Python script with the arguments
			python3 RTMA_download_variable_and_instancewise.py "$VARIABLE" "$LEVEL" "$year" "$month" "$day" "$hour" "$DOWNLOAD_PATH" &
			
			# Check the number of background jobs
			while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
			  # Wait for any job to finish
			  wait -n	
			done
	      done
	    done
	  done
	done
done
echo "All downloads have been finished."
