for file in slurmout/ddp_train-*.out; do
    filename=$(basename "$file")
    job_id="${filename##ddp_train-}"
    job_id="${job_id%.out}"
    #echo "Checking $file with job_id=$job_id"
    if [ "$job_id" -ge 11784 ]; then
        #echo "==== $file ===="
        tail -n 3 "$file"
    fi
done