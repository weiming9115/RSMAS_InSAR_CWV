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

os.chdir('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/pic/')
file = np.loadtxt('rms_timeseriesResidual_ramp.txt',skiprows=4)
date_acqui = []
for t in range(file.shape[0]):
    date_acqui.append(str(file[t,0])[:8]) # times of acquisitions

os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_data/sfc/regrid_3km/')
files = sorted(glob('*grib2'))
# merge HRRR dataset
# get one example file to have the size of a single ifgram image
tmp = xr.open_dataset(files[-2],engine='cfgrib',
                            backend_kwargs=dict(filter_by_keys={'stepType': 'instant','typeOfLevel':'unknown'}))
pwat_tmp = tmp.pwat.sel(latitude=slice(axis_bound[0],axis_bound[1]),longitude=slice(axis_bound[2]+360,axis_bound[3]+360))

pwat_acqu = np.zeros((len(date_acqui),pwat_tmp.shape[0],pwat_tmp.shape[1])) # create the array of the right size
ps_acqu = np.copy(pwat_acqu)
date_record = [] # save date_acqui as datetime for xarray

hourUTC = files[0][15:17] # file UTC, e.g., 01,03

with open("/data2/willytsai/InSAR_HRRR/auto_framework/"+CASE_ID+"/HRRR_log.txt",'w') as f:
    for t,date_str in enumerate(date_acqui):
        matched_fid = 'hrrr.'+date_str+'.t'+hourUTC+'z.regrid3km.grib2'  # hrrr.20160714.t02z.regrid3km.grib2
        file_exist = np.size(np.intersect1d(np.array(files),matched_fid))>0 # return True or False
        date_record.append(datetime.strptime(matched_fid[5:18],'%Y%m%d.t%Hz'))

        if file_exist == True:
            try:
                print(matched_fid)
                ds = xr.open_dataset(matched_fid,engine='cfgrib',
                                backend_kwargs=dict(filter_by_keys={'stepType': 'instant','typeOfLevel':'unknown'}))
                pwat = ds.pwat.sel(latitude=slice(axis_bound[0],axis_bound[1]),longitude=slice(axis_bound[2]+360,axis_bound[3]+360))

                pwat_acqu[t,:,:] = pwat.values

                ds = xr.open_dataset(matched_fid,engine='cfgrib',
                                backend_kwargs=dict(filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'}))
                ps = ds.sp.sel(latitude=slice(axis_bound[0],axis_bound[1]),longitude=slice(axis_bound[2]+360,axis_bound[3]+360))
                ps_acqu[t,:,:] = ps.values

            except:
                f.write('ERROR file: '+matched_fid)
                f.write('\n')
                pwat_acqu[t,:,:] = np.nan
                ps_acqu[t,:,:] = np.nan
        else: # if no file on GCP...
            f.write('MISSING file: '+matched_fid)
            f.write('\n')
            pwat_acqu[t,:,:] = np.nan
            ps_acqu[t,:,:] = np.nan

# convert into xarray and save as the netCDF format
pwat_acqu_xr = xr.DataArray(pwat_acqu,dims=('time','latitude','longitude')
                            ,coords=(date_record,pwat.latitude,pwat.longitude-360),name='pwat')
pwat_ds = pwat_acqu_xr.to_dataset(name='pwat')
os.system('rm -f /data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_pwat.nc')
pwat_ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_pwat.nc')

ps_acqu_xr = xr.DataArray(ps_acqu,dims=('time','latitude','longitude')
                            ,coords=(date_record,ps.latitude,ps.longitude-360),name='ps')
ps_ds = ps_acqu_xr.to_dataset(name='pressure')
os.system('rm -f /data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_ps.nc')
ps_ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_ps.nc')


