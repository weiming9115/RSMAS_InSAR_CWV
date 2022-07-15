#!/bin/bash

input="./batch_jobs.template"
while read -r line
do
   echo "CASE_ID = $line"
   python InSAR_ZTD_2netCDF_noIncidence.py $line
   python process_gps_insar_new.py $line
   echo "  "
   sleep 10
done < "$input"
