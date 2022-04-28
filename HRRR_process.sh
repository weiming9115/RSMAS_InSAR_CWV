#!/bin/bash

# CASE ID
CASE_ID=SouthWesternNAMChunk35SenAT64

#1.Create directories
echo "CASE_ID: " $CASE_ID
mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$CASE_ID
mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$CASE_ID/HRRR_data
mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$CASE_ID/HRRR_data/sfc
mkdir -p /data2/willytsai/InSAR_HRRR/auto_framework/$CASE_ID/NCEP_CPC_IRmerge

# ============ Application arguments =============
# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
    "-help") set -- "$@" "-h";;
    "-start") set -- "$@" "-s";;
    "-download") set --"$@" "-d";;
    "-start-no-download") set -- "$@" "-n";;
    *)        set -- "$@" "-w";;
  esac
done
#=================================================

# Parse short options
OPTIND=1
while getopts ":hsdnw" arg; do
  case $arg in
    # general operatins
    "h") echo "help";;
    "s") echo "HRRR_process: Regular start "
         #Download HRRR at InSAR acquistions
         echo "Downloading HRRR data from Google Cloud Platform" &&
	 python HRRR_Download_GCPlatform.py $CASE_ID             &&
         echo "Processing HRRR netcdfs as InSAR inputs"          &&
         python HRRR_vars_netcdf_ouput.py $CASE_ID               &&
         echo "Done HRRR part" ;;
    "d") #Download HRRR at InSAR acquistions
         echo "Downloading HRRR data from Google Cloud Platform" &&
         python HRRR_Download_GCPlatform.py $CASE_ID             &&
         echo "Download finished";;                              
    "n") echo "HRRR_process: No download"                        &&
         echo "Processing HRRR netcdfs as InSAR inputs"          &&
         python HRRR_vars_netcdf_ouput.py $CASE_ID;;             

    # invalid arguments
    "w") echo "Invalid option";;
  esac
done
shift $((OPTIND - 1)) # remove options from positional parameters

