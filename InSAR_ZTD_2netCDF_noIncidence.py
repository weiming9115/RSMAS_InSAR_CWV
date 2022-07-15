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
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
CASE_ID = sys.argv[1]
# get geolocation from InSAR
geo_file = '/data2/willytsai/InSAR_HRRR/data_Falk/'+CASE_ID+'/mintpy/inputs/geometryRadar.h5'
geo = h5py.File(geo_file,'r')
# for key in geo.keys():
#     print(key) #Names of the groups in HDF5 file.
lat = geo['latitude'];
lon = geo['longitude'];
incidence = geo['incidenceAngle'];
axis_bound = [np.min(lat),np.max(lat),np.min(lon),np.max(lon)]; # coordinate bound [South,North,West,East]
axis_bound = [np.unique(lat.value)[1],np.unique(lat.value)[-1],np.unique(lon.value)[0],np.unique(lon.value)[-2]]

# read ifgrams
#data_ifgrams = h5py.File('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/S1_IW123_166_0121_0140_20150322_XXXXXXXX.he5', 'r')
S1_file = glob('/data2/willytsai/InSAR_HRRR/data_Falk/'+CASE_ID+'/mintpy/S1_*')[0]
print('S1_file: ', S1_file)
data_ifgrams = h5py.File(S1_file, 'r')
geo_info = data_ifgrams['HDFEOS']['GRIDS']['timeseries']['geometry']
ifgrams = data_ifgrams['HDFEOS']['GRIDS']['timeseries']['observation']

inc_angle = geo_info['incidenceAngle']
disp_0 = ifgrams['displacement']
lon_ifgrams = geo_info['longitude']
lat_ifgrams = geo_info['latitude']
shadowMask = geo_info['shadowMask']
date_ifgrams = ifgrams['date']
height =  geo_info['height']

os.chdir('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_data/sfc/regrid_3km/')
files = sorted(glob('*grib2'))
hourUTC = files[0][15:17] # file UTC, e.g., 01,03
print('startUTC: ', hourUTC)

# generate time index
date = []
for t in date_ifgrams:
    date_full = t.decode("utf-8")+hourUTC
    date.append(datetime.strptime(date_full,'%Y%m%d%H'))

# flipping disp for ascending lat
disp = np.zeros(disp_0.shape)
for t in range(len(date_ifgrams)):
    disp[t,:,:] = -np.flipud(disp_0[t,:,:]) # delay singal [m]

lon_sar = np.linspace(axis_bound[2],axis_bound[3],lon_ifgrams.shape[1])
lat_sar = np.linspace(axis_bound[0],axis_bound[1],lon_ifgrams.shape[0])

#write insar into netcdf
disp_acqu_xr = xr.DataArray(disp,dims=('time','latitude','longitude')
                             ,coords=(date,lat_sar,lon_sar),name='displacement')
disp_acqu_ds = disp_acqu_xr.to_dataset(name='disp_zenith')
disp_acqu_ds.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/InSAR_zenithdisp.nc')
print('OUTFILE: InSAR_zenithdisp.nc')
print('InSAR ifgrams h5 to netCDF completed!')
