#!/bin/bash

input="./batch_jobs.template"
while read -r line
do
   echo "CASE_ID = $line"
   python NEXRAD_eth_refl_netcdf.py $line
   echo "  "
   sleep 10
done < "$input"
