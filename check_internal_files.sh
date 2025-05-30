#!/bin/bash

# Set the root folder path and target file count
root_folder="/data/harish/RTMA/TMP/rtma"  # Update this to the actual root folder path
target_file_count=24

# Loop through each subfolder in the root folder
for folder in "$root_folder"/*; do
    if [ -d "$folder" ]; then
        # Count the number of files in the subfolder (ignores hidden files)
        file_count=$(find "$folder" -type f | wc -l)
        
        # Check if the file count is not equal to the target file count
        if [ "$file_count" -ne "$target_file_count" ]; then
            echo "$(basename "$folder")"
        fi
    fi
done

