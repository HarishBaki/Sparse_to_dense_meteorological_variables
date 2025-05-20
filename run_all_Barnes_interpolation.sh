#!/bin/bash

# === Variable names (external and internal) ===

stations=(50 75 100)
var_names=("t2m" "si10")

for var_name in "${var_names[@]}"; do
    # Set mode: 'w' for i10fg (first write), 'a' for all others
    if [[ "$var_name" == "i10fg" ]]; then
        mode="w"
    else
        mode="a"
    fi
    for n_staitons in "${stations[@]}"; do
        echo "Processing variable: $var_name, n_stations $n_staitons, with mode: $mode"
        # Example: run test script
        python Barnes_interpolation.py $var_name $n_staitons $mode
        python Barnes_interpolation_NYSM.py $var_name $n_staitons $mode  
    done
    echo "Variable $var_name processed."
done