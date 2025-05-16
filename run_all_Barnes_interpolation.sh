#!/bin/bash

# === Variable names (external and internal) ===

stations=("none" 50 75 100)
var_names=("i10fg" "sh2" "t2m" "si10")

for var_name in "${var_names[@]}"; do
    # Set mode: 'w' for i10fg (first write), 'a' for all others
    if [[ "$var_name" == "i10fg" ]]; then
        mode="w"
        for n_staitons in "${stations[@]}"; do
            echo "Processing variable: $var_name, n_stations $n_staitons, with mode: $mode"
            # Example: run test script
            python Barnes_interpolation_NYSM.py $var_name $n_staitons $mode  
        done
    else
        mode="a"
        n_staitons='none'
        echo "Processing variable: $var_name, n_stations $n_staitons, with mode: $mode"
        # Example: run test script
        python Barnes_interpolation_NYSM.py $var_name $n_staitons $mode
    fi
    echo "Variable $var_name processed."
done