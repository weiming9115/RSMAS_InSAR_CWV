###################################################
           RSMAS InSAR-PW autoprocessing 
###################################################

1. Run "HRRR_process.sh -start" 
       one directory with the given name in .sh  will be generated, with 
       netcdf outputs including:
       -- HRRR_regrid3km_ps.nc
       -- HRRR_regrid3km_pwat.nc 
       -- InSAR_zenithdisp.nc

       arguments:
       -start: regular processing 
       -download: only download HRRR 
       -start-no-download: skip HRRR download and continue processing

       --------------------
       HRRR_log.txt: recording missing and error files

2. Run "process_insar_main.sh"
       main code for precipitable water reconstruction that generates an 
       output called "InSAR_derived_PWV_F.nc".
    
