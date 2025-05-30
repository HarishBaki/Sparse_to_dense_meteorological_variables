# === Variable names (external and internal) ===

# External names (folder-level or processing-level names)
variables=("GUST" "DPT" "PRES" "SPFH" "TMP" "WDIR" "WIND")
# Corresponding internal variable names (used inside individual NetCDF/Zarr files)
var_names=("i10fg" "d2m" "sp" "sh2" "t2m" "wdir10" "si10")

for i in "${!variables[@]}"; do
    variable="${variables[$i]}"
    var_name="${var_names[$i]}"

    # Set mode: 'w' for i10fg (first write), 'a' for all others
    if [[ "$variable" == "GUST" ]]; then
        mode="w"
    else
        mode="a"
    fi
    
    echo "Processing variable: $variable (internal name: $var_name), with mode: $mode"
    
    # Example: run test script
    python process_and_write_to_zarr.py $variable $mode $var_name 
done

echo "All variables processed."