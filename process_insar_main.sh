#!/bin/bash

CASE_ID=SouthWesternNAMChunk32SenAT20
echo "CASE_ID: " $CASE_ID
echo "InSAR-PW processing"

# check if HRRR data already prepared
INPUT_DIR=/data2/willytsai/InSAR_HRRR/auto_framework/$CASE_ID
F1=$INPUT_DIR"/HRRR_regrid3km_pwat.nc"
F2=$INPUT_DIR"/HRRR_regrid3km_ps.nc"

if [ -f $F1 -a -f $F2 ]; then
    echo "HRRR input files exist"
else
    echo "ERROR: HRRR input files missing"   
    exit 1
fi

# prepare InSAR inputs: zenith total delay
echo "Generate InSAR ZTD from S1-file"
python InSAR_ZTD_2netCDF.py $CASE_ID
# processing the main code
echo "Run InSAR-PW reconstruction"
python process_gps_insar.py $CASE_ID
