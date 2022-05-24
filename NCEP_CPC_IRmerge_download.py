#!/usr/bin/env python

# ## NCEP_CPC_IRmerge 
# Download GPM_IRmerge brightness temperature data from GESDISC archive

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cfgrib
import cf2cdm
from glob import glob
import xarray as xr
from datetime import datetime, timedelta
import urllib.request
from cfgrib.xarray_store import open_dataset
import warnings
import h5py
import time
from sys import argv

warnings.filterwarnings('ignore')

#################################
CASE_ID = argv[1]
#SAR_geometry = argv[2]
print('CASE_ID = ',CASE_ID)
#print('SAR_geometry = ',SAR_geometry)
#################################

os.chdir('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/pic/')
file = np.loadtxt('rms_timeseriesResidual_ramp.txt',skiprows=4)
date_acqui = []
for t in range(file.shape[0]):
    date_acqui.append(str(file[t,0])[:8]) # times of acquisitions
print('InSAR acquisitions:')

## get close UTC of Sentinel-1
s1_file = glob('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/*.he5')[0]
s1_he5 = h5py.File(s1_file,'r')
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

os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/NCEP_CPC_IRmerge')
for date in date_acqui:
    print(date)
    yr = date[:4]
    mon = date[4:6]
    day = date[6:8]
    
    day_in_yr = time.strptime(date, "%Y%m%d").tm_yday # convert into number of day in a year
    if day_in_yr < 10:
        day_folder = '00' + str(day_in_yr)
    elif (day_in_yr >= 10) and (day_in_yr < 100):
        day_folder = '0' + str(day_in_yr)
    else:
        day_folder = str(day_in_yr)
    
    cmmd = str('wget --user=willyqq9115 --password=As@23082606 https://disc2.gesdisc.eosdis.nasa.gov/data/MERGED_IR/GPM_MERGIR.1/'+ 
            yr +'/'+ day_folder + '/merg_' + date + hh + '_4km-pixel.nc4')
    os.system(cmmd)

# select domain
geo_file = '/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/inputs/geometryRadar.h5'
geo = h5py.File(geo_file,'r')
# for key in geo.keys():
#     print(key) #Names of the groups in HDF5 file.
lat = geo['latitude'];
lon = geo['longitude'];
incidence = geo['incidenceAngle'];
axis_bound = [np.min(lat),np.max(lat),np.min(lon),np.max(lon)]; # coordinate bound [South,North,West,East]
axis_bound = [np.unique(lat.value)[1],np.unique(lat.value)[-1],np.unique(lon.value)[0],np.unique(lon.value)[-2]]
axis_bound

# have some IR information from NCEP_CPC_IRmerge
os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/NCEP_CPC_IRmerge/')
IR_files = sorted(glob('*.nc4'))
data_ir = xr.open_dataset(IR_files[0])
Tb = data_ir.sel(lat=slice(axis_bound[0],axis_bound[1]),lon=slice(axis_bound[2],axis_bound[3])).Tb

data_ir = xr.open_dataset(IR_files[0])
tmp = data_ir.sel(lat=slice(axis_bound[0],axis_bound[1]),lon=slice(axis_bound[2],axis_bound[3])).Tb
# array for output
Tb_acqu = np.empty((len(IR_files),len(tmp.lat),len(tmp.lon)))

date_frame = []
for n, file in enumerate(IR_files):
    date_frame.append(datetime.strptime(file[5:15],'%Y%m%d%H'))

    data_ir = xr.open_dataset(file)
    Tb = data_ir.sel(lat=slice(axis_bound[0],axis_bound[1]),lon=slice(axis_bound[2],axis_bound[3])).Tb
    Tb_out = Tb[0,:,:]
    Tb_acqu[n,:,:] = Tb_out

# finish timeseries of Tb for all acquisitions
Tb_acqu_xr = xr.DataArray(Tb_acqu,dims=('time','latitude','longitude')
                            ,coords=(date_frame,Tb.lat,Tb.lon),name='Tb')
Tb_ds = Tb_acqu_xr.to_dataset(name='Tb')
# output direcotry and file name
Tb_ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/NCEP_CPC_IRmerge/NCEP_CPC_Tb_'+CASE_ID+'.nc')


