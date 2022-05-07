#!/bin/bash

input="./batch_jobs.template"
while read -r line
do
   echo "CASE_ID = $line"
   python process_gps_insar.py $line
   echo "  "
   sleep 10
done < "$input"
