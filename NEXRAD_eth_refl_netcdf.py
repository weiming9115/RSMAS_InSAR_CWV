### Calculating composite reflectivity and echo top height
#reference<br>
#Lakshmanan et al. (2013), "An Improved Method for Estimating Radar Echo-Top Height". Weather Forecast. 28, 481–488,doi:10.1175/WAF-D-12-00084.1.

#codes:
#https://github.com/vlouf/eth_radar/blob/master/echotop/echotop.py

import sys
import matplotlib.pyplot as plt
import tempfile
import pytz
from datetime import datetime, date, time, timedelta
import pyart
import numpy as np
from mpl_toolkits.basemap import Basemap
import h5py
from glob import glob
templocation = tempfile.mkdtemp()

import pandas as pd
import warnings
import cartopy.crs as ccrs
import xarray as xr

from scipy.interpolate import interp2d, griddata

warnings.filterwarnings('ignore')

import nexradaws
conn = nexradaws.NexradAwsInterface()

from multiprocessing import Pool
from echotop import cloud_top_height, grid_cloud_top, column_max_reflectivity

def radar_selection(case_id, case_date):
    """find radar data enclosed in the InSAR scene"""
    
    geo_file = '/data2/willytsai/InSAR_HRRR/data_Falk/'+case_id+'/mintpy/inputs/geometryRadar.h5'
    geo = h5py.File(geo_file,'r')
    lat = geo['latitude'];
    lon = geo['longitude'];
    axis_bound = [np.min(lat),np.max(lat),np.min(lon),np.max(lon)]; # coordinate bound [South,North,West,East]
    axis_bound = [np.unique(lat[()])[1],np.unique(lat[()])[-1],np.unique(lon[()])[0],np.unique(lon[()])[-2]]
    
    s1_file = glob('/data2/willytsai/InSAR_HRRR/data_Falk/'+case_id+'/mintpy/*.he5')[0]
    s1_he5 = h5py.File(s1_file,'r')
    startUTC = s1_he5.attrs['startUTC']
    stopUTC = s1_he5.attrs['stopUTC']
    
    df = pd.read_pickle('/data2/willytsai/InSAR_HRRR/auto_framework/NEXRAD_latlon.pkl')
    df_NEXRAD = df.loc[(df['lat']>axis_bound[0]) & (df['lat']<axis_bound[1]) 
               & (df['lon']>axis_bound[2]) & (df['lon']<axis_bound[3])]
    lat_center = 0.5*(axis_bound[0]+axis_bound[1])
    lon_center = 0.5*(axis_bound[2]+axis_bound[3])
    df_NEXRAD['distance_cent'] = ((df_NEXRAD['lon']-lon_center)**2 + (df_NEXRAD['lat']-lat_center)**2)**1/2
    df_NEXRAD = df_NEXRAD.sort_values('distance_cent')
    
    # generate timestamp 
    _date = date(case_date.year,case_date.month,case_date.day)
    _time = time(int(startUTC[11:13]),int(startUTC[14:16]))
    timestamp_utc = datetime.combine(_date,_time, pytz.UTC)
    timestamp = datetime.combine(_date,_time)
    
    return df_NEXRAD, timestamp_utc, timestamp, axis_bound

def eth_refl_calc(dtime):
    
    timestamp = ((dtime.values - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    case_time = datetime.utcfromtimestamp(timestamp)
    
    pwv_insar = data.sel(time=dtime,method='nearest').pwat
    
    (df_NEXRAD, start_timeUTC, start_time, axis_bound) = radar_selection(case_id, case_time)

    central_timezone = pytz.timezone('US/Central')
    radar_id = df_NEXRAD.index[0] # radar station which is closest to the domain center
    start = central_timezone.fromutc(start_time - timedelta(minutes=10)) # 30mins before InSAR scene
    end = central_timezone.fromutc(start_time + timedelta(minutes=10)) # 30mins after InSAR scene
    
    
    try:
        scans = conn.get_avail_scans_in_range(start, end, radar_id)

        # find the nearest radar observaiton matching the InSAR acquisition, setting it at the ceneter of subplots
        scan_time_list = []
        for scan in scans:
            tt_diff = (scan.scan_time - start_timeUTC).total_seconds()
            scan_time_list.append(tt_diff)
        scan_time_list = np.asarray(scan_time_list)

        # calculating only if observations exist...
        if len(scan_time_list) > 0: 
            scan_time_square = scan_time_list**2 # choose the minimux
            idx_sel = np.argsort(scan_time_square)[0]

            results = conn.download(scans[idx_sel], templocation)

            for i,scan in enumerate(results.iter_success(),start=0):
                    radar = scan.open_pyart()

                    # ================== calculation starts ==================
                    # Reading data.
                    r = radar.range['data']
                    refl = radar.fields['reflectivity']['data'].copy()
                    azimuth = radar.azimuth['data']
                    elevation = radar.elevation['data']

                    sw = radar.get_slice(0)    
                    refl_ref = refl[sw]
                    azi_ref = radar.azimuth['data'][sw]
                    azi_level_0 = radar.azimuth['data'][sw]    
                    elev_ref = radar.elevation['data'][sw].mean()

                    x_radar, y_radar, _ = radar.get_gate_x_y_z(0)

                    R, A = np.meshgrid(r, azi_level_0)    
                    try:
                        zhrad = radar.altitude['data'].mean()
                    except Exception:
                        zhrad = 50 

                    st_sweep = radar.sweep_start_ray_index['data']
                    ed_sweep = radar.sweep_end_ray_index['data']

                    # Theta 3dB angle.
                    da = azi_level_0[1] - azi_level_0[0]    
                    if np.abs(1 - da) < np.abs(1.5 - da):
                        th3 = 1
                    else:
                        th3 = 1.5

                    refl_compo = column_max_reflectivity(r, azimuth, elevation, st_sweep, ed_sweep, refl)
                    cloudtop = eth = cloud_top_height(r, azimuth, elevation, st_sweep, ed_sweep, refl, eth_thld=10) # thresold 10 dbz

                    #============== gridding process ================
                    refc = radar.get_field(0,'reflectivity')
                    (lat,lon,elev) = radar.get_gate_lat_lon_alt(0)

                    lon_new = pwv_insar.longitude
                    lat_new = pwv_insar.latitude
                    grid_lon, grid_lat = np.meshgrid(lon_new, lat_new) # grids for interpolation into InSAR grids

                    lon_1d = lon.ravel()
                    lat_1d = lat.ravel()
                    points = np.stack([lon_1d,lat_1d]).swapaxes(0,1) # points for lat, lon in radar sweep_num=0

                    refl_compo[refc.mask==1] = np.nan # applying mask 
                    refl_compo_1d = refl_compo.ravel()
                    cloudtop[refc.mask==1] = np.nan
                    cloudtop_1d = cloudtop.ravel()

                    _refl_compo_grid = griddata(points, refl_compo_1d, (grid_lon, grid_lat), method='nearest')
                    _eth_grid = griddata(points, cloudtop_1d, (grid_lon, grid_lat), method='nearest')

        else:
            # if no matching radar observations...
            _refl_compo_grid = pwv_insar*np.nan
            _eth_grid = pwv_insar*np.nan
    
    except:
            _refl_compo_grid = pwv_insar*np.nan
            _eth_grid = pwv_insar*np.nan
            
    return _refl_compo_grid, _eth_grid

case_id =  sys.argv[1]
pwv_dir = '/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/outfiles/'
data = xr.open_dataset(pwv_dir+'InSAR_derived_PWV_F.nc') # InSAR PWV data

# output files: composite reflectivity, echo top height (10dBZ)
refl_compo_grid = np.empty(data.pwat.shape)
eth_grid = np.copy(refl_compo_grid)

process_pool = Pool()

out_results = process_pool.map(eth_refl_calc, data.time, chunksize=10)

for i in range(len(out_results)):
    refl_compo_grid[i,:,:] = out_results[i][0]
    eth_grid[i,:,:] = out_results[i][1]
    
process_pool.close()

# save into xarray 
refl_compo_xr = xr.DataArray(refl_compo_grid, coords=(data.time, data.latitude, data.longitude),
                            dims=('time','latitude','longitude'),name='refc_composite')
eth_xr = xr.DataArray(eth_grid, coords=(data.time, data.latitude, data.longitude),
                            dims=('time','latitude','longitude'),name='eth')

# save netcdf files
ds1 = refl_compo_xr.to_dataset(name='refc_composite')
ds1.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/NEXRAD_refc_composite.nc')
ds2 = eth_xr.to_dataset(name='eth')
ds2.to_netcdf('/data2/willytsai/InSAR_HRRR/auto_framework/'+case_id+'/NEXRAD_echotopheight_10dBZ.nc')
