#!/bin/bash

input="./batch_jobs.template"
while read -r line
do
   echo "CASE_ID = $line"
   mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$line
   mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$line/HRRR_data
   mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$line/HRRR_data/sfc
   mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$line/NCEP_CPC_IRmerge

   echo "------------------"
   echo "Downloading HRRR data from Google Cloud Platform"
   python HRRR_Download_GCPlatform.py $line
   echo "Processing HRRR netcdfs as InSAR inputs"
   python HRRR_vars_netcdf_ouput.py $line
   echo "Generate InSAR ZTD from S1-file"
   python InSAR_ZTD_2netCDF_noIncidence.py $line
   echo "Run InSAR-PW reconstruction"
   python process_gps_insar_new.py $line
   echo "------------------  "
   sleep 10
done < "$input"
