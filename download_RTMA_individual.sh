#!/bin/bash

# Variables to be passed to Python script
VARIABLE="GUST"
LEVEL="10 m"
DOWNLOAD_PATH="/data/harish"

# Array of dates in yearmonthday format
dates=("20181105" "20190918" "20191208" "20200125" "20200307" "20200308" "20200414" "20200415" "20200521" "20200522" "20200523" "20200524" "20200525" "20200526" "20200625" "20201112")

# Loop through each date
for date in "${dates[@]}"; do
    # Extract year, month, and day using substring
    year=${date:0:4}
    month=${date:4:2}
    day=${date:6:2}
    
    for hour in {00..23}; do
	# Call the Python script with the arguments
	python3 RTMA_download_variable_and_instancewise.py "$VARIABLE" "$LEVEL" "$year" "$month" "$day" "$hour" "$DOWNLOAD_PATH" &
    done
    wait
done

echo "All downloads have been finished."
