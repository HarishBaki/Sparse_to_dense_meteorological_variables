VARIABLES=("WDIR" "SPFH") 
for i in "${!VARIABLES[@]}"; do
	VARIABLE="${VARIABLES[i]}"
    rclone copy --progress --transfers 8 "/data/harish/RTMA/$VARIABLE" "harish_NAS:/data/RTMA/$VARIABLE"
done
echo "All transfers have been finished."