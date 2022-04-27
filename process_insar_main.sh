#!/bin/bash

CASE_ID=SouthWesternNAMChunk28SenAT166
echo "CASE_ID: " $CASE_ID
echo "InSAR-PW processing"
# prepare InSAR inputs: zenith total delay
echo "Generate InSAR ZTD from S1-file"
python InSAR_ZTD_2netCDF.py $CASE_ID
# processing the main code
echo "Run InSAR-PW reconstruction"
python process_gps_insar.py $CASE_ID
