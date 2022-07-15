import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
#import cfgrib
#import cf2cdm
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime, timedelta
import urllib.request
#from cfgrib.xarray_store import open_dataset
import warnings
import h5py
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression


import scipy.signal as signal
import scipy.fft as fft
from skimage.feature import match_template

warnings.filterwarnings('ignore')

def remap_hrrr2SAR(hrrr_2dmap):
    "regrid hrrr data into SAR"
    lon_hrrr,lat_hrrr = np.meshgrid(hrrr_2dmap.longitude.values,hrrr_2dmap.latitude.values)
    lon_isar,lat_isar = np.meshgrid(lon_sar,lat_sar)
    points = np.stack([lon_hrrr.ravel(),lat_hrrr.ravel()]).swapaxes(0,1)
    var_remap = griddata(points, hrrr_2dmap.values.ravel(), (lon_isar, lat_isar), method='linear')

    return var_remap

case_id = sys.argv[1]
file_src_insar = '/data2/willytsai/InSAR_HRRR/data_Falk/'+case_id+'/mintpy/'
file_dir_insar = '/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/outfiles/'
file_dir_hrrr = '/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/'
file_dir_CPC = '/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/NCEP_CPC_IRmerge/'

pwv_insar_abs = xr.open_dataset(file_dir_insar + 'InSAR_derived_PWV_F.nc')
pwv_hrrr_abs = xr.open_dataset(file_dir_hrrr+'HRRR_regrid3km_pwat.nc')

# load GPS station info
data_GPS = pd.read_pickle(file_dir_insar+'data_GPSinfo.pkl')

# get geolocation from InSAR
geo_file = '/data2/willytsai/InSAR_HRRR/data_Falk/'+case_id+'/mintpy/inputs/geometryRadar.h5'
geo = h5py.File(geo_file,'r')
# for key in geo.keys():
#     print(key) #Names of the groups in HDF5 file.
lat = geo['latitude'];
lon = geo['longitude'];
incidence = geo['incidenceAngle'];
axis_bound = [np.min(lat),np.max(lat),np.min(lon),np.max(lon)]; # coordinate bound [South,North,West,East]
axis_bound = [np.unique(lat.value)[1],np.unique(lat.value)[-1],np.unique(lon.value)[0],np.unique(lon.value)[-2]]

# the reference point on InSAR map, near P002
S1_file =glob(file_src_insar+'S1_*')[0]
data_ifgrams = h5py.File(S1_file, 'r')
geo_info = data_ifgrams['HDFEOS']['GRIDS']['timeseries']['geometry']
disp = data_ifgrams['HDFEOS']['GRIDS']['timeseries']['observation']['displacement']
lon_ifgrams = geo_info['longitude']
lat_ifgrams = geo_info['latitude']
elev = geo_info['height']

lon_sar = np.linspace(axis_bound[2],axis_bound[3],lon_ifgrams.shape[1])
lat_sar = np.linspace(axis_bound[0],axis_bound[1],lon_ifgrams.shape[0])

# calculating velocity for mask
velocity = np.zeros((disp.value.shape[1], disp.value.shape[2]))
disp_array = np.fliplr(disp.value)

datetime_secs = []
for dtime in pwv_hrrr_abs.time:
    
    timestamp = ((dtime.values - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    case_time = datetime.utcfromtimestamp(timestamp)
    
    datetime_secs.append((case_time-datetime(1970,1,1)).total_seconds())

disp_xr = xr.DataArray(disp_array,
                       coords=(datetime_secs, pwv_insar_abs.latitude, pwv_insar_abs.longitude),
                       dims=('time','latitude','longitude'))
tmp = disp_xr.polyfit(dim='time',deg=1)
velocity = tmp.polyfit_coefficients[0,:,:]*86400*365*100 # cm/year

# remapping HRRR-PW to match subswathes
pwv_hrrr_remap = np.empty(pwv_insar_abs.pwat.shape)

for t in range(len(pwv_hrrr_abs.time)):
    pwv_hrrr_remap[t,:,:] = remap_hrrr2SAR(pwv_hrrr_abs.pwat[t,:,:])
pwv_hrrr_remap = xr.DataArray(pwv_hrrr_remap, 
                              coords = [pwv_insar_abs.time, pwv_insar_abs.latitude, pwv_insar_abs.longitude],
                              dims = ['time','latitude','longitude'],name='pwat')
pwv_hrrr_remap = pwv_hrrr_remap.to_dataset(name='pwat')

#### matching swathes ####
pwv_hrrr_swath = (pwv_hrrr_remap).where(pwv_insar_abs>0)

pwv_hrrr_mask = pwv_hrrr_swath.pwat[t,:,:].values
pwv_hrrr_mask[abs(velocity.values)> 0.5] = np.nan

pwv_insar_mask = pwv_insar_abs.pwat[t,:,:].values
pwv_insar_mask[abs(velocity.values)> 0.5] = np.nan

data = xr.open_dataset(file_dir_hrrr+'NEXRAD_refc_composite.nc')
refc_NEX = data.refc_composite

data = xr.open_dataset(file_dir_hrrr+'NEXRAD_refc_composite_hrrr.nc')
refc_NEX_hrrr = data.refc_composite

data = xr.open_dataset(file_dir_hrrr+'HRRR_regrid3km_refc.nc')
refc_hrrr = data.refc

idx_conv = np.where(refc_NEX.max(('longitude','latitude'))>30)[0]
idx_conv

refc_mask = refc_NEX[t,:,:].values
refc_mask[abs(velocity.values)> 0.5] = np.nan

refc_hrrr_mask = refc_NEX_hrrr[t,:,:].values
refc_hrrr_mask[abs(velocity.values)> 0.5] = np.nan

# boxplots for pwv-p 
pzr_bins_hrrr_hist = []
pzr_bins_insar_hist = []

pwv_insar_hist = np.empty(1)
pwv_hrrr_hist = np.empty(1)
refc_insar_hist = np.empty(1)
refc_hrrr_hist = np.empty(1)

pwv_bins = np.linspace(15,65,11)


# for InSAR
idx_conv_insar = np.where(refc_NEX.max(('longitude','latitude'))>30)[0]
idx_conv_insar

for n,t in enumerate(idx_conv_insar): # for CONV times
        
    pwv_insar_mask = pwv_insar_abs.pwat[t,:,:].values # pwv at 01Z
    pwv_insar_mask[abs(velocity.values)> 0.5] = np.nan
    refc_insar_mask = refc_NEX[t,:,:].values # refc NEX at 01Z
    refc_insar_mask[abs(velocity.values)> 0.5] = np.nan
    
    z = np.power(10,refc_insar_mask/10)
    P_zr_insar = np.power((z/300),1/1.4) # mm/hr
            
    idx = np.isfinite(pwv_insar_mask[refc_insar_mask>0]) & np.isfinite(P_zr_insar[refc_insar_mask>0])
    tmp1 = pwv_insar_mask[refc_insar_mask>0][idx]
    tmp2 = P_zr_insar[refc_insar_mask>0][idx]
    tmp3 = refc_insar_mask[refc_insar_mask>0][idx]    

    if n == 0:
        pwv_insar_hist = tmp1
        refc_insar_hist = tmp3
    else:
        pwv_insar_hist = np.concatenate((pwv_insar_hist,tmp1))
        refc_insar_hist = np.concatenate((refc_insar_hist,tmp3))
    
    for i in range(len(pwv_bins)-1):
        
        idx_com = np.where(np.logical_and( tmp1 > pwv_bins[i], tmp1 <= pwv_bins[i+1]))
        
        if n == 0: # first data into list
            pzr_bins_insar_hist.append(tmp2[idx_com])
        else:
            pzr_bins_insar_hist[i] = np.concatenate((pzr_bins_insar_hist[i],tmp2[idx_com]))
            
    
# for InSAR
idx_conv_hrrr = np.where(refc_NEX_hrrr.max(('longitude','latitude'))>30)[0]
idx_conv_hrrr

for n,t in enumerate(idx_conv_hrrr): # for CONV times
    
    pwv_hrrr_mask = pwv_hrrr_swath.pwat[t,:,:].values # pwv at 01Z
    pwv_hrrr_mask[abs(velocity.values)> 0.5] = np.nan
    refc_hrrr_mask = refc_NEX_hrrr[t,:,:].values # refc NEX at 01Z
    refc_hrrr_mask[abs(velocity.values)> 0.5] = np.nan
    
    z = np.power(10,refc_hrrr_mask/10)
    P_zr_hrrr = np.power((z/300),1/1.4) # mm/hr
    
    idx = np.isfinite(pwv_hrrr_mask[refc_hrrr_mask>0]) & np.isfinite(P_zr_hrrr[refc_hrrr_mask>0])
    tmp1 = pwv_hrrr_mask[refc_hrrr_mask>0][idx]
    tmp2 = P_zr_hrrr[refc_hrrr_mask>0][idx]
    tmp3 = refc_hrrr_mask[refc_hrrr_mask>0][idx]
    
    if n == 0:
        pwv_hrrr_hist = tmp1
        refc_hrrr_hist = tmp3
    else:
        pwv_hrrr_hist = np.concatenate((pwv_hrrr_hist,tmp1))
        refc_hrrr_hist = np.concatenate((refc_hrrr_hist,tmp3))

    for i in range(len(pwv_bins)-1):
        
        idx_com = np.where(np.logical_and( tmp1 > pwv_bins[i], tmp1 <= pwv_bins[i+1]))
        
        if n == 0: # first data into list
            pzr_bins_hrrr_hist.append(tmp2[idx_com])
        else:
            pzr_bins_hrrr_hist[i] = np.concatenate((pzr_bins_hrrr_hist[i],tmp2[idx_com]))

np.save(file_dir_hrrr+'pzr_bins_hrrr_hist_max30dbz',pzr_bins_hrrr_hist)
np.save(file_dir_hrrr+'pzr_bins_insar_hist_max30dbz',pzr_bins_insar_hist)
np.save(file_dir_hrrr+'pwv_hrrr_hist_max30dbz',pwv_hrrr_hist)
np.save(file_dir_hrrr+'pwv_insar_hist_max30dbz',pwv_insar_hist)
np.save(file_dir_hrrr+'refc_hrrr_hist_max30dbz',refc_hrrr_hist)
np.save(file_dir_hrrr+'refc_insar_hist_max30dbz',refc_insar_hist)
np.save(file_dir_hrrr+'scenes_num_max30dbz',np.array([len(idx_conv_insar),len(idx_conv_hrrr)]))





