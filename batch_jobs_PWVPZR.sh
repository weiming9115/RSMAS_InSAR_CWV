#!/bin/bash

input="./batch_jobs.template"
while read -r line
do
   echo "CASE_ID = $line"
   python InSAR_pwv_pzr_out.py $line
   echo "  "
   sleep 10
done < "$input"
