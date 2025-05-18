#!/bin/bash
squeue -u $USER --noheader --format="%i %j %T" | awk '$2=="metrics_test" && $3=="PENDING" {print $1}' | while read jobid; do
    echo "Cancelling pending job ID: $jobid"
    scancel "$jobid"
done

squeue -u $USER --noheader --format="%i %j %T" | awk '$2=="metrics_test" && $3=="RUNNING" {print $1}' | while read jobid; do
    echo "Cancelling running job ID: $jobid"
    scancel "$jobid"
done