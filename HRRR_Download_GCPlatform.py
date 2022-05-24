#!/usr/bin/env python

# ### Download HRRR data from Google Cloud Platform
# Wei-Ming Tsai, Feb. 2022 <br>
# HRRR data archive: https://console.cloud.google.com/storage/browser/high-resolution-rapid-refresh
# how to use GSP API: gsutil cp gs://high-resolution-rapid-refresh/hrrr.20140730/conus/hrrr.t18z.wrfsfcf06.grib2 .

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cfgrib
import cf2cdm
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import xarray as xr
from datetime import datetime, timedelta
import urllib.request
from cfgrib.xarray_store import open_dataset
import warnings
import h5py

warnings.filterwarnings('ignore')
case_id = sys.argv[1]
os.chdir('/data2/willytsai/InSAR_HRRR/'+case_id+'/mintpy/pic/')
file = np.loadtxt('rms_timeseriesResidual_ramp.txt',skiprows=4)
date_acqui = []
for t in range(file.shape[0]):
    date_acqui.append(str(file[t,0])[:8]) # times of acquisitions

# ### Download HRRR data from archive
os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/HRRR_data/')
os.system('mkdir -p sfc')
os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/HRRR_data/sfc/')

## get close UTC of Sentinel-1
s1_file = glob('/data2/willytsai/InSAR_HRRR/'+case_id+'/mintpy/*.he5')[0]
s1_he5 = h5py.File(s1_file,'r')
print('satellite time: ',(s1_he5.attrs['startUTC'])
hh = s1_he5.attrs['startUTC'][11:13] # hr 
mm = s1_he5.attrs['startUTC'][14:16] # mm
if int(mm) > 30:
    hh = str(int(hh)+1)
    hh = str(hh).zfill(2)
    if int(hh) >= 24:
        hh = int(hh)-24
        hh = str(hh).zfill(2) # convert 0 to 00 or 1 to 01 ,etc
        if hh == '00':
            hh ='23' # for simplicity
print('startUTC = ', hh)

# Downloading best analyzed HRRR output at specfied UTC.
####################################################
for date in date_acqui:
    for hour in [hh]:
        file_name = 'hrrr.'+date+'.t'+hour+'z.grib2'
        cmmd1 = 'gsutil cp gs://high-resolution-rapid-refresh/hrrr.'+date+'/conus/hrrr.t'+hour+'z.wrfsfcf00.grib2 .'
        cmmd2 = 'mv '+'hrrr.t'+hour+'z.wrfsfcf00.grib2'+' '+ file_name 
        os.system(cmmd1)
        os.system(cmmd2)
####################################################

# Regrid HRRR data 
os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/HRRR_data/sfc/')
os.system('mkdir -p regrid_3km')
files = sorted(glob('*grib2'))
cmd = '/home/willytsai/grib2/wgrib2/wgrib2'
attr = '-set_grib_type same -new_grid_winds earth -new_grid latlon 225.9:2440:0.03 21.14:1049:0.03'
for file in files:
    input_file = file
    output_file = file[:19]+'regrid3km'+'.grib2'
    os.system(cmd+' '+input_file+' '+attr+' ./regrid_3km/'+output_file)

