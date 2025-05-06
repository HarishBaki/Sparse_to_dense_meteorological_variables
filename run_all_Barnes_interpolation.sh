#!/bin/bash

# === Variable names (external and internal) ===

stations=("none" 50 75 100)
var_names=("i10fg" "sh2" "t2m" "si10")
for n_staitons in "${stations[@]}"; do
    for i in "${!var_names[@]}"; do
        var_name="${var_names[$i]}"

        # Set mode: 'w' for i10fg (first write), 'a' for all others
        if [[ "$var_name" == "i10fg" ]]; then
            mode="w"
        else
            mode="a"
        fi
        
        echo "Processing variable: $var_name, n_stations $n_staitons, with mode: $mode"
        
        # Example: run test script
        python Barnes_interpolation.py $var_name $n_staitons $mode  
    done

    echo "All variables processed."

done