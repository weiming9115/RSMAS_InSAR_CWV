#!/usr/bin/env python
# Process regridded, equirectangularly projected, HRRR variable into merged netcdf

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

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.stats import linregress

warnings.filterwarnings('ignore')

CASE_ID = sys.argv[1]
# get geolocation from InSAR
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

os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_data/sfc/regrid_3km/')
files = sorted(glob('*grib2'))

# merge HRRR dataset 
# get one example file to have the size of a single ifgram image
tmp = xr.open_dataset(files[-1],engine='cfgrib',
                            backend_kwargs=dict(filter_by_keys={'stepType': 'instant','typeOfLevel':'unknown'}))
pwat_tmp = tmp.pwat.sel(latitude=slice(axis_bound[0],axis_bound[1]),longitude=slice(axis_bound[2]+360,axis_bound[3]+360))

# giving the acquisition info, that is, time
os.chdir('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/pic/')
file = np.loadtxt('rms_timeseriesResidual_ramp.txt',skiprows=4)
date_acqui = []
for t in range(file.shape[0]):
    date_acqui.append(str(file[t,0])[:8]) # times of acquisitions

from multiprocessing import Pool
from functools import partial

def merge_vars(file_id, var_name, typeoflevel):
    
    try:
        ds = xr.open_dataset(file_id,engine='cfgrib',
                            backend_kwargs=dict(filter_by_keys={'stepType': 'instant','typeOfLevel':typeoflevel}))
        var = ds[var_name].sel(latitude=slice(axis_bound[0],axis_bound[1]),longitude=slice(axis_bound[2]+360,axis_bound[3]+360))
        var_acqu = var.values   
    except:
        print('ERROR file: '+file_id)
        var_acqu = np.nan*np.zeros((pwat_tmp.shape[0],pwat_tmp.shape[1]))
        
    return var_acqu

def merge_vars_layer(file_id, var_name, typeoflevel):

    try:
        ds = xr.open_dataset(file_id,engine='cfgrib',
                            backend_kwargs=dict(filter_by_keys={'stepType': 'instant','typeOfLevel':typeoflevel}))
        var = ds[var_name].sel(latitude=slice(axis_bound[0],axis_bound[1]),longitude=slice(axis_bound[2]+360,axis_bound[3]+360))
        var_acqu = var[1,:,:].values # here is 925mb layer for u, v wind
    except:
        print('ERROR file: '+file_id)
        var_acqu = np.nan*np.zeros((pwat_tmp.shape[0],pwat_tmp.shape[1]))

    return var_acqu

def timestamp_write(file_id):
    timestamp = datetime.strptime(file_id[5:18],'%Y%m%d.t%Hz')
    return timestamp

# rm existing old files
os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID)
os.system('rm -f HRRR_regrid*') # remove HRRR_regrid3km_xxx.nc 

# merge HRRR dataset 
## get close UTC of Sentinel-1
s1_file = glob('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/*.he5')[0]
s1_he5 = h5py.File(s1_file,'r')
print('satellite time: ',(s1_he5.attrs['startUTC']))
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

os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_data/sfc/regrid_3km/')
file_list = []
for x in date_acqui:
    file_list.append('hrrr.'+x+'.t'+hh+'z.regrid3km.grib2')  # expected files at acquisitions
# merge HRRR dataset
# get one example file to have the size of a single ifgram image
tmp = xr.open_dataset(files[-2],engine='cfgrib',
                            backend_kwargs=dict(filter_by_keys={'stepType': 'instant','typeOfLevel':'unknown'}))
pwat_tmp = tmp.pwat.sel(latitude=slice(axis_bound[0],axis_bound[1]),longitude=slice(axis_bound[2]+360,axis_bound[3]+360))

# main merging process -- unknown layer
var_list = ['pwat']
var_label = ['pwat']

for var_name,var_label in zip(var_list,var_label):

    process_pool = Pool()    

    # write timestamp for xarray
    date_frame = process_pool.map(timestamp_write, file_list, chunksize=10)
    
    # start processing each variable
    process_var = partial(merge_vars, var_name = var_name, typeoflevel='unknown') 
    da = process_pool.map(process_var, file_list, chunksize=10) # data merge
    da = np.asarray(da)
    # convert data into xarray 
    da_acqu_xr = xr.DataArray(da,dims=('time','latitude','longitude')
                            ,coords=(date_frame,pwat_tmp.latitude,pwat_tmp.longitude-360),name=var_name)
    ds = da_acqu_xr.to_dataset(name=var_name)
    ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_'+var_label+'.nc')

    process_pool.close()

# main merging process -- surface layer
var_list = ['sp','prate','gust','t'] # surface pressure, rainfall, gust wind, surface temperature
var_label = ['ps','prate','gust','Ts']

for var_name,var_label in zip(var_list,var_label):

    process_pool = Pool()

    # write timestamp for xarray
    date_frame = process_pool.map(timestamp_write, file_list, chunksize=10)
    
    # start processing each variable
    process_var = partial(merge_vars, var_name = var_name, typeoflevel='surface') 
    da = process_pool.map(process_var, file_list, chunksize=10) # data merge
    da = np.asarray(da)
    # convert data into xarray 
    da_acqu_xr = xr.DataArray(da,dims=('time','latitude','longitude')
                            ,coords=(date_frame,pwat_tmp.latitude,pwat_tmp.longitude-360),name=var_name)
    ds = da_acqu_xr.to_dataset(name=var_name)
    ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_'+var_label+'.nc')
    
    process_pool.close()

# main merging process -- atmosphere layer
var_list = ['refc']
var_label = ['refc']

for var_name,var_label in zip(var_list,var_label):

    process_pool = Pool()

    # write timestamp for xarray
    date_frame = process_pool.map(timestamp_write, file_list, chunksize=10)

    # start processing each variable
    process_var = partial(merge_vars, var_name = var_name, typeoflevel='atmosphere')
    da = process_pool.map(process_var, file_list, chunksize=10) # data merge
    da = np.asarray(da)
    # convert data into xarray
    da_acqu_xr = xr.DataArray(da,dims=('time','latitude','longitude')
                            ,coords=(date_frame,pwat_tmp.latitude,pwat_tmp.longitude-360),name=var_name)
    ds = da_acqu_xr.to_dataset(name=var_name)
    ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_'+var_label+'.nc')

    process_pool.close()

# main merging process -- atmosphere layer
var_list = ['u','v']
var_label = ['u925mb','v925mb']

for var_name,var_label in zip(var_list,var_label):

    process_pool = Pool()

    # write timestamp for xarray
    date_frame = process_pool.map(timestamp_write, file_list, chunksize=10)

    # start processing each variable
    process_var = partial(merge_vars_layer, var_name = var_name, typeoflevel='isobaricInhPa')
    da = process_pool.map(process_var, file_list, chunksize=10) # data merge
    da = np.asarray(da)
    # convert data into xarray
    da_acqu_xr = xr.DataArray(da,dims=('time','latitude','longitude')
                            ,coords=(date_frame,pwat_tmp.latitude,pwat_tmp.longitude-360),name=var_name)
    ds = da_acqu_xr.to_dataset(name=var_name)
    ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_'+var_label+'.nc')

    process_pool.close()

print('done HRRR data merge')

