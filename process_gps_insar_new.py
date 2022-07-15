#=============================================
#   InSAR Precipitable Water Reconstruction
#=============================================
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cfgrib
import cf2cdm
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime, timedelta
import urllib.request
from cfgrib.xarray_store import open_dataset
import warnings
import h5py
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

CASE_ID = sys.argv[1]

#============ PARAMETERS =============
geo_file = '/data2/willytsai/InSAR_HRRR/data_Falk/'+CASE_ID+'/mintpy/inputs/geometryRadar.h5'
date_ref = np.loadtxt('/data2/willytsai/InSAR_HRRR/data_Falk/'+CASE_ID+'/mintpy/pic/reference_date.txt')

OUTDIR = '/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/outfiles/'

#=====================================
os.system('rm -rf '+OUTDIR) # remove old folder
os.system('mkdir -p '+OUTDIR) # create OUTDIR

#============ FUNCTIONS ============
#============ FUNCTIONS =============
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def merge_pkl_files(var_name, year):
    # var_name = variable name
    # year = [2015,2016..] list
    data_merged = pd.DataFrame()
    for yr in year:
        #tmp = pd.read_pickle('/data2/willytsai/InSAR_HRRR/COSMIC_GPSPWV/'+str(yr)+'/data_'+var_name+'.pkl')
        tmp = pd.read_pickle('/data2/willytsai/InSAR_HRRR/COSMIC_GPS_ARM/ftp.archive.arm.gov/'+str(yr)+'/data_'+var_name+'.pkl')
        data_merged = data_merged.append(tmp, sort=False)
    return data_merged

def station_acqu_subdomain(stat_df, axis_bound):
    # extract GPS data covered by the specified region

    stat_sub = stat_df.loc[stat_df['lat'].between(axis_bound[0],axis_bound[1]) &
               stat_df['lon'].between(axis_bound[2],axis_bound[3])]

    return stat_sub

def remap_hrrr2SAR(hrrr_2dmap):
    "regrid hrrr data into SAR"
    lon_hrrr,lat_hrrr = np.meshgrid(hrrr_2dmap.longitude.values,hrrr_2dmap.latitude.values)
    lon_isar,lat_isar = np.meshgrid(lon_sar,lat_sar)
    points = np.stack([lon_hrrr.ravel(),lat_hrrr.ravel()]).swapaxes(0,1)
    var_remap = griddata(points, hrrr_2dmap.values.ravel(), (lon_isar, lat_isar), method='linear')

    return var_remap

def find_latlon_ref(lon_ifgrams, lat_ifgrams):

    ifg_file = '/data2/willytsai/InSAR_HRRR/data_Falk/'+CASE_ID+'/mintpy/inputs/ifgramStack.h5'
    ifg_file = h5py.File(ifg_file,'r')
    REF_X = int(ifg_file.attrs['REF_X'])
    REF_Y = int(ifg_file.attrs['REF_Y'])

    lon_ref = lon[REF_Y,REF_X] # lat of reference point
    lat_ref = lat[REF_Y,REF_X] # lon of reference point

    return lon_ref, lat_ref

def nearest_station_Ref(station_id_subdomain, lon_ref, lat_ref):

    station_id_subdomain['dist2ref'] = np.sqrt((station_id_subdomain['lon'] - lon_ref)**2 +
                                           (station_id_subdomain['lat'] - lat_ref)**2)
    try:
        return (station_id_subdomain.sort_values(by='dist2ref').head(1).index.values[0],
                station_id_subdomain.sort_values(by='dist2ref').head(1).lon.values[0],
                station_id_subdomain.sort_values(by='dist2ref').head(1).lat.values[0])
    except:
        return str('None')

def best_station_Ref(tot_delay_sub, tot_delay_insar, station_id_subdomain, lon_ref, lat_ref, datetime_ref):
    ''' best GPS as our new reference point.
        (1) must have value at datetime_ref
        (2) contains longest valid records'''

    valid_ref = np.zeros(len(station_id_subdomain))
    valid_record = np.copy(valid_ref)

    for n,stat in enumerate(station_id_subdomain.index.values[:]):
        if np.isnan(tot_delay_sub[stat].loc[datetime_ref]) == 0:
            valid_ref[n] = 1
        valid_record[n] = np.sum(np.isnan(tot_delay_sub[stat])) # number of values not NaN

    station_id_subdomain['valid_ref'] = valid_ref
    station_id_subdomain['valid_record'] = np.array(valid_record,dtype=int)
    # extract valid stations
    tmp = station_id_subdomain.where(station_id_subdomain['valid_ref']>=1)
    data_gps_reloc = tmp.sort_values(by='valid_record',ascending=False) # sorting stations by valid records
    
    # check if GPS station is out of InSAR imagery
    for n,stat in enumerate(data_gps_reloc.index):
        lon = data_gps_reloc.loc[stat].lon
        lat = data_gps_reloc.loc[stat].lat
        tmp = tot_delay_insar.sel(longitude=lon, latitude=lat, method='nearest')
        
        if np.sum(np.isnan(tmp.values)) != len(tmp.time):
            
            return (data_gps_reloc.index.values[n],
                    data_gps_reloc.lon.values[n],
                    data_gps_reloc.lat.values[n],data_gps_reloc)
        
        else:
            continue

def view_domain(delay_insar_2d, station_id_subdomain, lon_refnew, lat_refnew):
    "return fig containing enclosed GPS sites and ZTD at the acquisition time"

    fig,ax = plt.subplots(1,1,figsize=(8,8),subplot_kw={'projection': ccrs.PlateCarree()})

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes_shp',
        scale='50m',
        facecolor='none')

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(states_provinces, edgecolor='gray')

    data_showcase_ZTD = delay_insar_2d
    cf = ax.contourf(data.longitude,data.latitude,data_showcase_ZTD)
    cbar = plt.colorbar(cf,ax=ax,shrink=0.4); cbar.set_label('ZTD [mm]')
    ax.set_title(str(data_showcase_ZTD.time.values)[:-10], fontweight='bold', fontsize=13)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle=':')
    gl.xlabels_top=False; gl.ylabels_right=False

    ax.scatter(station_id_subdomain.lon,station_id_subdomain.lat,color='orange',marker='^')
    ax.plot(lon_ref,lat_ref,'x',color='r')
    ax.plot(lon_refnew,lat_refnew,'o',color='m')
    for stat in station_id_subdomain.index:
        ax.annotate(stat,(station_id_subdomain.loc[stat].lon,station_id_subdomain.loc[stat].lat+0.1))

    return fig

## testing, relocate the default InSAR reference point to one existing GPS station
def reset_tot_delay_ref(tot_delay_insar, tot_delay_sub, lon_ref, lat_ref, station_id_subdomain, datetime_ref):
    '''relocate the defualt reference point to a GPS station. The origional InSAR inteferogram at
       one acqusition will be added a constant(offest) which makes the value at new default reference
       point zero, while the relative values between each pixels remain the same!'''

    (ref_statID, lon_refnew, lat_refnew, data_gps_reloc) = best_station_Ref(tot_delay_sub, tot_delay_insar
                                                            ,station_id_subdomain, lon_ref, lat_ref, datetime_ref)

    delay_ref_stat = tot_delay_insar.sel(longitude=lon_refnew, latitude=lat_refnew, method='nearest') # 1D timeseries at new ref (GPS station)
    delay_ref_insar = tot_delay_insar.sel(longitude=lon_ref, latitude=lat_ref, method='nearest') # 1D timeseries at insar default ref.
    offset = delay_ref_insar - delay_ref_stat # offset at each time

    # add offset that makes delay at GPS become zero, the new reference point
    tot_delay_insar_reloc = tot_delay_insar + offset

    return ( tot_delay_insar_reloc, ref_statID )

def view_ZTDdiff_relocate(tot_delay_sub_insar, tot_delay_sub, station_id_subdomain, datetime_ref):

    fig = plt.figure(figsize=(6,6))
    for i in station_id_subdomain.index.values[:]:
        tmp = tot_delay_sub_insar[i] # insar total delay diff [mm]
        tmp2 = tot_delay_sub[i]-tot_delay_sub.loc[datetime_ref][i] # COSMIC total delay diff

        plt.scatter(tmp,tmp2,marker='^',label=i)
        plt.ylabel('COSMIC ZTD diff. [mm]',fontsize=13);plt.xlabel('InSAR ZTD diff. [mm]',fontsize=13)
        plt.legend()
    return fig

def plot_ZTDoffset_series(offset_best):
    ''' plot the best ZTD offset blended by GPS and HRRR. Sometimes GPS is not valid, so we use HRRR
        as substitution at the new reference point. This allows to maximize the value of InSAR data we collect...'''

    fig = plt.figure(figsize=(10,3))
    plt.plot(offset,'ob')
    plt.plot(offset_byhrrr,'-k')
    plt.plot(offset_best,'-r')
    plt.title('Valid GPS Records: '+str(valid_record)+' out of '
          + str(len(tot_delay_sub_insar[ref_statID])),fontsize=14,loc='right')
    plt.legend(['offset-GPS','offset-HRRR','offset-best'],frameon=False)
    plt.ylabel('ZTD offset at REF [mm]',fontsize=12)
    plt.grid(linestyle=':',linewidth=0.5)
    return fig

#=============== END FUNCTIONS ===============

#======================================================
#  SECT. A. Preparation datasets for InSAR-CWV retrival
#======================================================

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
print('Box boundaries: ',axis_bound)
date_ref = np.loadtxt('/data2/willytsai/InSAR_HRRR/data_Falk/'+CASE_ID+'/mintpy/pic/reference_date.txt')
print('Reference time: ',date_ref)

# 0. Get InSAR data (processed by rsmas_insar software)
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/InSAR_zenithdisp.nc')
data = data.sel(time=slice(data.time[0],data.time[-1])) 
year_start = int(data.time[0].values.astype(str).split('-')[0]) # year of the first acquisition
year_end = int(data.time[-1].values.astype(str).split('-')[0]) # year of the last acquisition

year_list = list(range(year_start,year_end+1)) # create the list of years for GPS data

# 1. Get GPS data (processed by COSMIC_ProcessDataframe.py) at each acquisition
stat_df = merge_pkl_files(var_name='statInfo',year=year_list)
# remove duplicates in station index (same name but slightly different values in lon lat...)
stat_df = stat_df[~stat_df.index.duplicated(keep='first')]

pwv_df = merge_pkl_files(var_name='pwv',year=year_list)
wet_delay_df = merge_pkl_files(var_name='wet_delay',year=year_list)
dry_delay_df = merge_pkl_files(var_name='final_dry_delay',year=year_list)
pifactor_df = merge_pkl_files(var_name='pifact',year=year_list)

# Data quality control: removing missing values in GPS
pwv_df[pwv_df<0] = np.nan
wet_delay_df[wet_delay_df<0] = np.nan
dry_delay_df[dry_delay_df<0] = np.nan
pifactor_df[pifactor_df<0] = np.nan

# extract t02z data over CONUS stations to match the acquisition time of InSAR
date_list = []
for date in data.time.values[data.time.values < np.datetime64(datetime(year_end,12,31,0))]:
    date_list.append(datetime.utcfromtimestamp(date.astype(int) * 1e-9))
pwv_acqu = pwv_df.loc[pwv_df.index.intersection(date_list),:]
wet_delay_acqu = wet_delay_df.loc[wet_delay_df.index.intersection(date_list),:]
dry_delay_acqu = dry_delay_df.loc[dry_delay_df.index.intersection(date_list),:]
tot_delay_acqu = wet_delay_acqu + dry_delay_acqu
pifactor_acqu = pifactor_df.loc[pifactor_df.index.intersection(date_list),:]

# show GPS stations within the box boundaries
station_id_subdomain = station_acqu_subdomain(stat_df, axis_bound)
station_id_subdomain.to_pickle(OUTDIR+'data_GPSinfo.pkl')
print('OUTFILE: data_GPSinfo.pkl')

#------- station-wise comparison between GPS and InSAR zenith delay -------
# return reference gps station ID for later adjustment
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/InSAR_zenithdisp.nc')
tot_delay_insar = data.disp_zenith*1000 # origional insar data [mm]
tot_delay_sub = tot_delay_acqu[station_id_subdomain.index] # gps data

hrUTC = str(tot_delay_sub.index[0].hour).zfill(2)
datetime_ref = datetime.strptime(str(date_ref)[:-2]+hrUTC,'%Y%m%d%H')
(lon_ref, lat_ref) = find_latlon_ref(lon,lat)

#--------------------------------------------------------------------------
# selecting subdomain dataframes
tot_delay_sub = tot_delay_acqu[station_id_subdomain.index] # gps
pwv_sub = pwv_acqu[station_id_subdomain.index]             # gps
pifactor_sub = pifactor_acqu[station_id_subdomain.index]   # gps

tot_delay_sub_insar = pd.DataFrame()
for i in station_id_subdomain.index.values:

    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = tot_delay_insar.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'disp_zenith': i})
    tot_delay_sub_insar = pd.concat([tot_delay_sub_insar,tmp],axis=1)

# 2. Adding back the offset removed in ZTD interferograms by GPS
hrUTC = str(tot_delay_sub.index[0].hour).zfill(2)
datetime_ref = datetime.strptime(str(date_ref)[:-2]+hrUTC,'%Y%m%d%H')
print('Origional reference point:', lon_ref, lat_ref)
print('Reference datetime: ', datetime_ref)

fig = view_domain(tot_delay_insar[6,:,:],station_id_subdomain, lon_ref, lat_ref)
fig.savefig(OUTDIR+'InSAR_GPS_view.png',dpi=1200,bbox_inches='tight',transparent=False)
print('OUTFILE: InSAR_GPS_view.png')
plt.close()

# =========================================================
#  SECT. B start InSAR-CWV retrival if ref_statID exists
#  some swathes with ref point over seas... or no GPS nearby
# =========================================================

# the reference point on InSAR map
S1_file = glob('/data2/willytsai/InSAR_HRRR/data_Falk/'+CASE_ID+'/mintpy/S1_*.he5')[0]
data_ifgrams = h5py.File(S1_file, 'r')
geo_info = data_ifgrams['HDFEOS']['GRIDS']['timeseries']['geometry']
lon_ifgrams = geo_info['longitude']
lat_ifgrams = geo_info['latitude']

lon_sar = np.linspace(axis_bound[2],axis_bound[3],lon_ifgrams.shape[1])
lat_sar = np.linspace(axis_bound[0],axis_bound[1],lon_ifgrams.shape[0])

# creating total_delay_hrrr to replace GPS if needed... try to fully used our InSAR images
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_ps.nc')
ps_acqu = data.sp
ps_diff = ps_acqu-ps_acqu.sel(time=datetime_ref)
hydro_factor = 1e-6*0.776*287.15/9.8 # [m/pa]
sar_ps_diff = np.zeros((len(date_list),len(lat_sar),len(lon_sar)))

for t in range(sar_ps_diff.shape[0]):
    sar_ps_diff[t,:,:] = remap_hrrr2SAR(ps_diff[t,:,:]) # regrid pressure diff into InSAR grid

sar_dry_delay = hydro_factor*sar_ps_diff #[m/pa]*[pa]
sar_dry_delay = xr.DataArray(sar_dry_delay,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='dry_delay')
sar_dry_delay = sar_dry_delay.to_dataset(name='dry_delay')

# generate the 3-D dry delay field from HRRR surface pressure outputs
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_ps.nc')
ps_acqu = data.sp
ps_diff = ps_acqu-ps_acqu.sel(time=datetime_ref)
hydro_factor = 1e-6*0.776*287.15/9.8 # [m/pa]
sar_ps_diff = np.zeros((len(date_list),len(lat_sar),len(lon_sar)))

for t in range(len(date_list)):
    sar_ps_diff[t,:,:] = remap_hrrr2SAR(ps_diff[t,:,:]) # regrid pressure diff into sar grid

dry_delay_diff_hrrr = hydro_factor*sar_ps_diff #[m/pa]*[pa]
dry_delay_diff_hrrr = xr.DataArray(dry_delay_diff_hrrr,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='dry_delay_diff')
dry_delay_diff_hrrr = dry_delay_diff_hrrr.to_dataset(name='dry_delay_diff')

# adding HRRR ZWD_diff to compensate for missing GPS times at the refenrence point
pwv_hrrr = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_pwat.nc')
pwv_hrrr = pwv_hrrr.sel(time=slice(datetime(year_start,1,1),datetime(year_end,12,31)))
pwv_diff_hrrr = pwv_hrrr - pwv_hrrr.sel(time=datetime_ref) # pwv_diff to converted into wet_delay_diff_hrrr

wet_delay_diff_hrrr = np.copy(tot_delay_insar)*np.nan

for t in range(len(date_list)):
    try:
        wet_delay_diff_hrrr[t,:,:] = remap_hrrr2SAR(pwv_diff_hrrr.pwat[t,:,:]*6.3)
    except:
        continue
wet_delay_diff_hrrr = xr.DataArray(wet_delay_diff_hrrr,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='wet_delay_diff')
wet_delay_diff_hrrr = wet_delay_diff_hrrr.to_dataset(name='wet_delay_diff')

tot_delay_diff_hrrr = xr.DataArray((dry_delay_diff_hrrr['dry_delay_diff'].values + wet_delay_diff_hrrr['wet_delay_diff'].values),dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='wet_delay_diff')
tot_delay_diff_hrrr = tot_delay_diff_hrrr.to_dataset(name='tot_delay_diff')

# building ZTD_diff_InSAR, ZTD_diff_HRRR and ZTD_diff_gps at GPS stations

tot_delay_diff_sub_hrrr = pd.DataFrame()

for i in station_id_subdomain.index.values:

    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    # find nearest point close to GPS stations
    tmp = tot_delay_diff_hrrr['tot_delay_diff'].sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'tot_delay_diff': i})
    tot_delay_diff_sub_hrrr = pd.concat([tot_delay_diff_sub_hrrr,tmp],axis=1)

tot_delay_diff_sub_insar = pd.DataFrame()

for i in station_id_subdomain.index.values:

    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    # find nearest point close to GPS stations
    tmp = tot_delay_insar.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'disp_zenith': i})
    tot_delay_diff_sub_insar = pd.concat([tot_delay_diff_sub_insar,tmp],axis=1)
    
tot_delay_diff_sub_gps = tot_delay_sub - tot_delay_sub.loc[datetime_ref]

# blending ZTD from GPS and HRRR for our best benchmark 
tot_delay_diff_sub_best = tot_delay_diff_sub_gps.copy()

for stat in station_id_subdomain.index:
    tmp = tot_delay_diff_sub_gps[stat].values
    for t in range(len(date_list)):
        if np.isnan(tmp[t]) == 1: # if no gps signals 
            tot_delay_diff_sub_best.iloc[t][stat] = tot_delay_diff_sub_hrrr.iloc[t][stat]

## ZTD differnece comparison
fig,ax = plt.subplots(3,2,figsize=(15,7))

ncols = [0,1,0,1,0,1]
nrows = [0,0,1,1,2,2]

for n,stat in enumerate(station_id_subdomain.index[:6]):
    lon_tmp = station_id_subdomain.lon[n]
    lat_tmp = station_id_subdomain.lat[n]
    # GPS ZTD - ZTD_ref
    ax[nrows[n],ncols[n]].plot(tot_delay_insar.time,
                               tot_delay_diff_sub_gps[stat]
                               ,'ro') # ZTD [mm]
    # insar ZTD at GPS point
    ax[nrows[n],ncols[n]].plot(tot_delay_insar.time,tot_delay_diff_sub_insar[stat],'k') # ZTD [mm]
    ax[nrows[n],ncols[n]].set_ylabel('$\Delta$ZTD [mm]',fontsize=13)
    ax[nrows[n],ncols[n]].set_title(stat,loc='right',fontsize=14,fontweight='bold')
    
    # hrrr ZTD at GPS point
    ax[nrows[n],ncols[n]].plot(tot_delay_insar.time,tot_delay_diff_sub_hrrr[stat],'b') # ZTD [mm]
    ax[nrows[n],ncols[n]].set_ylabel('$\Delta$ZTD [mm]',fontsize=13)
    ax[nrows[n],ncols[n]].set_title(stat,loc='right',fontsize=14,fontweight='bold')    
    
    if n == 0:
        ax[nrows[n],ncols[n]].legend(['GPS','InSAR','HRRR'],loc=2)
    
    ax[nrows[n],ncols[n]].plot(datetime_ref,0,'s',color='green') # reference time 
    ax[nrows[n],ncols[n]].grid(linestyle=':',linewidth=0.5)
    
plt.tight_layout()
    
fig.savefig(OUTDIR+'ZTDdiff_rawdata_comparison.pdf',
            bbox_inches='tight',transparent=False)
plt.close()

# InSAR calibration: building cost-function to add constants, which were arbitrarily introdueced 
# in the unwrapping process, into InSAR timeseies (Mateus et al, 2012)

offset_best = np.empty(len(tot_delay_diff_sub_best.index))

#######################################
K = np.linspace(-200,400,1200) # offset is usually positve [mm]
#######################################

for t in range(len(tot_delay_diff_sub_best.index)):
    cost_func = np.empty(len(K))*np.nan # storing cos_func calculated
    
    for n,k in enumerate(K): 
        tmp = 0 # reset cost

        for stat in station_id_subdomain.index: # suming from all stations enclosed, only available if values exist        
            if (np.isnan(tot_delay_diff_sub_best.iloc[t][stat]) != 1) & (np.isnan(tot_delay_diff_sub_insar.iloc[t][stat]) != 1):     
                tmp += (tot_delay_diff_sub_best.iloc[t][stat]-(tot_delay_diff_sub_insar.iloc[t][stat] + k))**2
        cost_func[n] = tmp
    
    cost_func[np.isnan(cost_func)] = 1e9 # giving a huge number if NaN due to missing data
    try:
        idx_sel = np.where(cost_func == np.min(cost_func))[0][0] # find the K 
        offset_best[t] = K[idx_sel] # k that minimizes the cost function
    except:
        offset_best[t] = np.nan
        
# calibrating InSAR ZTD
tot_delay_diff_sub_insarC = tot_delay_diff_sub_insar.copy()

for stat in station_id_subdomain.index:
    tot_delay_diff_sub_insarC[stat] = tot_delay_diff_sub_insar[stat] + offset_best

fig,ax = plt.subplots(3,2,figsize=(15,7))

ncols = [0,1,0,1,0,1]
nrows = [0,0,1,1,2,2]

for n,stat in enumerate(station_id_subdomain.index[:6]):
    lon_tmp = station_id_subdomain.lon[n]
    lat_tmp = station_id_subdomain.lat[n]
    # GPS ZTD - ZTD_ref
    ax[nrows[n],ncols[n]].plot(tot_delay_insar.time,
                               tot_delay_diff_sub_gps[stat]
                               ,'ro',markersize=4) # ZTD [mm]
    
    # blended ZTD at GPS points
    ax[nrows[n],ncols[n]].plot(tot_delay_insar.time,tot_delay_diff_sub_best[stat],'k',alpha=0.5) # ZTD [mm]
    ax[nrows[n],ncols[n]].set_ylabel('$\Delta$ZTD [mm]',fontsize=13)
    ax[nrows[n],ncols[n]].set_title(stat,loc='right',fontsize=14,fontweight='bold')  
    
    # calibrated InSAR ZTD at GPS points
    ax[nrows[n],ncols[n]].plot(tot_delay_insar.time,tot_delay_diff_sub_insarC[stat],'r',alpha=0.5) # ZTD [mm]
    ax[nrows[n],ncols[n]].set_ylabel('$\Delta$ZTD [mm]',fontsize=13)
    ax[nrows[n],ncols[n]].set_title(stat,loc='right',fontsize=14,fontweight='bold')  
    
    if n == 0:
        ax[nrows[n],ncols[n]].legend(['GPS','Blended','InSAR-Adj'],loc=2)
    
    ax[nrows[n],ncols[n]].plot(datetime_ref,0,'s',color='green') # reference time 
    ax[nrows[n],ncols[n]].grid(linestyle=':',linewidth=0.5)
    
plt.tight_layout()
    
fig.savefig(OUTDIR+'ZTDdiff_calibrated_comparison.pdf',
             bbox_inches='tight',transparent=False)
plt.close()

insar_rec = []; gps_rec = []

for stat in station_id_subdomain.index:
    tmp = tot_delay_diff_sub_insarC[stat] # insar total delay diff [mm]
    tmp2 = tot_delay_diff_sub_best[stat] # COSMIC total delay diff [mm]

    insar_rec.append(tmp.values)
    gps_rec.append(tmp2.values)

    plt.scatter(tmp,tmp2,marker='^',label=stat)
    
insar_rec = np.array(insar_rec).ravel()
gps_rec = np.array(gps_rec).ravel()

tmp, tmp2 = [],[]
for i in range(len(insar_rec)):
    if np.isnan(insar_rec[i]) == False and np.isnan(gps_rec[i]) == False:
        tmp.append(insar_rec[i])
        tmp2.append(gps_rec[i])

r_stats = linregress(tmp,tmp2)
plt.plot(np.linspace(-45,350,51),r_stats.slope*np.linspace(-45,350,51)+r_stats.intercept,
        color='k')
plt.plot(np.linspace(-45,350,51),1.0*np.linspace(-45,350,51),color='lightgrey')
plt.xlim([-45,350]);plt.ylim([-45,350])
plt.text(-10,300,'Corr: '+str(round(r_stats.rvalue,3)),fontsize=14)
plt.grid(linestyle=':',linewidth=0.5)
plt.ylabel('GPS+HRRR ZTD [mm]',fontsize=13);plt.xlabel('InSAR ZTD [mm]',fontsize=13)
plt.legend(loc=4)
fig.savefig(OUTDIR+'ZTD_addoffset_scatterplot.png',dpi=600,
           bbox_inches='tight',facecolor='white', transparent=False)
plt.close()

# =================================
#    4. Derive InSAR wet delay and PW
# =================================

# creating offset maps 
offset_2d = np.empty(tot_delay_insar.shape)
for t in range(len(date_list)):
    offset_2d[t,:,:] = offset_best[t]
offset_2d = xr.DataArray(offset_2d, dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='offset_best')

# estimate the zenith wet delay by combining InSAR and atmospheric vars from HRRR
wet_delay_diff_insar = (tot_delay_insar + offset_2d) - dry_delay_diff_hrrr.dry_delay_diff # [mm] 
wet_delay_diff_insar = wet_delay_diff_insar.to_dataset(name='wet_delay_diff')

# InSAR
wet_delay_diff_sub_insar = pd.DataFrame()

for stat in station_id_subdomain.index:

    lon_sel = station_id_subdomain.loc[stat]['lon']
    lat_sel = station_id_subdomain.loc[stat]['lat']
    # find nearest point close to GPS stations
    tmp = wet_delay_diff_insar.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'wet_delay_diff': stat})
    wet_delay_diff_sub_insar = pd.concat([wet_delay_diff_sub_insar,tmp],axis=1)
    
# HRRR
wet_delay_diff_sub_hrrr = pd.DataFrame()

for stat in station_id_subdomain.index:

    lon_sel = station_id_subdomain.loc[stat]['lon']
    lat_sel = station_id_subdomain.loc[stat]['lat']
    # find nearest point close to GPS stations
    tmp = wet_delay_diff_hrrr.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'wet_delay_diff': stat})
    wet_delay_diff_sub_hrrr = pd.concat([wet_delay_diff_sub_hrrr,tmp],axis=1)    

# GPS
# station-wise comparison between GPS PWV
wet_delay_sub = wet_delay_acqu[station_id_subdomain.index]
wet_delay_sub = wet_delay_sub.replace(-99.900002,np.nan)
wet_delay_diff_sub_gps = wet_delay_sub - wet_delay_sub.loc[datetime_ref]

# ZWD adjusted timeseries of InSAR and selected GPS stations -- see how well InSAR matches GPS
station_avail = station_id_subdomain.index.values[:6] # pick 6 stations, including the reference GPS
nrow = [0,0,1,1,2,2]
ncol = [0,1,0,1,0,1]

fig,ax = plt.subplots(3,2,figsize=(18,7))

for i,stat in enumerate(station_avail):

    ax[nrow[i],ncol[i]].plot(pwv_sub.index,wet_delay_diff_sub_insar[stat], 'or') # original InSAR ZWD diff
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,wet_delay_diff_sub_hrrr[stat],'grey',linewidth=1) # original HRRR ZWD diff
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,wet_delay_sub[stat]-wet_delay_sub.loc[datetime_ref][stat],'ob')
    loc_str = ('  ('+str(round(station_id_subdomain.loc[stat]['lon'],2))
               +', '+str(round(station_id_subdomain.loc[stat]['lat'],2)) + ')')
    ax[nrow[i],ncol[i]].set_title(stat + loc_str,fontsize=13, fontweight='bold')
    ax[nrow[i],ncol[i]].set_ylabel('ZWD_diff [mm]',fontsize=13)
    ax[nrow[i],ncol[i]].set_ylim([-40,120])

    ax[nrow[i],ncol[i]].grid(linestyle=':',linewidth=0.5)
    ax[nrow[i],ncol[i]].legend(['InSAR-Re','HRRR','COSMIC GPS'],loc=2)
    ax[nrow[i],ncol[i]].plot(datetime_ref,0,color='gold',marker='s')
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,np.zeros(len(pwv_sub.index)),
                             color='lightgrey')
    ax[nrow[i],ncol[i]].set_xlim([date_list[0],date_list[-1]])
    ax[nrow[i],ncol[i]].spines['right'].set_visible(False)
    ax[nrow[i],ncol[i]].spines['top'].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR+'ZWD_diff_timeseries.png',dpi=600,
           bbox_inches='tight',facecolor='white', transparent=False)
plt.close()

# remove offset to get relative ZWD reference to GPS and divided by pi_factor (6.3 here)
# to convert ZWD into PWV at the reference time and point
# =======================================================================================
#   PWV(x,t) = Pi*(ZDT(x,t) + offset(t)) + PWV(x,tref), following Mateus et al., 2012
# =======================================================================================
pwv_insar_abs = np.copy(wet_delay_diff_insar.wet_delay_diff)
pwv_hrrr_ref = pwv_hrrr.sel(time=datetime_ref).pwat
pwv_hrrr_ref_remap = remap_hrrr2SAR(pwv_hrrr_ref)

for i,t in enumerate(date_list):
    # PW_insar[t] = ZWD_insar[t]/6.3 + PW[t_ref]
    pwv_insar_abs[i,:,:] = wet_delay_diff_insar.wet_delay_diff[i,:,:]/6.3 + pwv_hrrr_ref_remap[:,:]
pwv_insar_abs = xr.DataArray(pwv_insar_abs,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='pwat')
pwv_insar_abs = pwv_insar_abs.to_dataset(name='pwat')

pwv_sub_hrrr = pd.DataFrame()
for stat in station_id_subdomain.index:

    lon_sel = station_id_subdomain.loc[stat]['lon']
    lat_sel = station_id_subdomain.loc[stat]['lat']
    tmp = pwv_hrrr.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'pwat': stat})
    pwv_sub_hrrr = pd.concat([pwv_sub_hrrr,tmp],axis=1)

pwv_sub_insar = pd.DataFrame()
for stat in station_id_subdomain.index:

    lon_sel = station_id_subdomain.loc[stat]['lon']
    lat_sel = station_id_subdomain.loc[stat]['lat']
    tmp = pwv_insar_abs.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'pwat': stat})
    pwv_sub_insar = pd.concat([pwv_sub_insar,tmp],axis=1)

# ZWD adjusted timeseries of InSAR and selected GPS stations -- see how well InSAR matches GPS
station_avail = station_id_subdomain.index.values[:6] # pick 6 stations, including the reference GPS
nrow = [0,0,1,1,2,2]
ncol = [0,1,0,1,0,1]

fig,ax = plt.subplots(3,2,figsize=(18,7))

for i,stat in enumerate(station_avail):

    ax[nrow[i],ncol[i]].plot(pwv_sub.index,pwv_sub_insar[stat], 'or') # InSAR-PW
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,pwv_sub_hrrr[stat],'grey',linewidth=1) # HRRR-PW
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,pwv_sub[stat],'ob')
    loc_str = ('  ('+str(round(station_id_subdomain.loc[stat]['lon'],2))
               +', '+str(round(station_id_subdomain.loc[stat]['lat'],2)) + ')')
    ax[nrow[i],ncol[i]].set_title(stat + loc_str,fontsize=13, fontweight='bold')
    ax[nrow[i],ncol[i]].set_ylabel('ZWD_diff [mm]',fontsize=13)
    ax[nrow[i],ncol[i]].set_ylim([0,55])

    ax[nrow[i],ncol[i]].grid(linestyle=':',linewidth=0.5)
    ax[nrow[i],ncol[i]].legend(['InSAR-Re','HRRR','COSMIC GPS'],loc=2)
    ax[nrow[i],ncol[i]].plot(datetime_ref,0,color='gold',marker='s')
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,np.zeros(len(pwv_sub.index)),
                             color='lightgrey')
    ax[nrow[i],ncol[i]].set_xlim([date_list[0],date_list[-1]])
    ax[nrow[i],ncol[i]].spines['right'].set_visible(False)
    ax[nrow[i],ncol[i]].spines['top'].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR+'PW_abs_timeseries.png',dpi=1200,
           bbox_inches='tight',facecolor='white', transparent=False)

#=============================================
#  5. Scatterplot of InSAR  PW at GPS staitons
#==============================================
fig,ax = plt.subplots(1,1,figsize=(5,5))

insar_rec = []; gps_rec = [];

for i in station_id_subdomain.index.values[:]:
    tmp = pwv_sub_insar[i] # insar derived PWV
    tmp2 = pwv_sub[i] # COSMIC PWV

    insar_rec.append(tmp.values)
    gps_rec.append(tmp2.values)

    plt.scatter(tmp,tmp2,marker='^',label=i)
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

insar_rec = np.array(insar_rec).ravel()
gps_rec = np.array(gps_rec).ravel()

tmp, tmp2 = [],[]
for i in range(len(insar_rec)):
    if np.isnan(insar_rec[i]) == False and np.isnan(gps_rec[i]) == False:
        tmp.append(insar_rec[i])
        tmp2.append(gps_rec[i])

r_stats = linregress(tmp,tmp2)
ax.plot(np.linspace(-30,30,51),r_stats.slope*np.linspace(-30,30,51)+r_stats.intercept,
        color='k')
ax.plot(np.linspace(-30,30,51),1.0*np.linspace(-30,30,51),color='lightgrey')
ax.set_xlim([0,30]);ax.set_ylim([0,30])
ax.text(20,25,'R_value:'+str(round(r_stats.rvalue,3)),fontsize=14)
ax.grid(linestyle=':',linewidth=0.5)
ax.set_ylabel('COSMIC PWV [mm]',fontsize=14);ax.set_xlabel('InSAR PWV [mm]',fontsize=14)
ax.legend()
fig.savefig(OUTDIR+'InSAR_GPS_PWV_scatterplot.png',dpi=1200,
          bbox_inches='tight',facecolor='white', transparent=False)
plt.close()

# HRRR vs GPS
fig,ax = plt.subplots(1,1,figsize=(5,5))

hrrr_rec = []; gps_rec = [];

for i in station_id_subdomain.index.values[:]:
    tmp = pwv_sub_hrrr[i] # insar derived PWV
    tmp2 = pwv_sub[i] # COSMIC PWV

    hrrr_rec.append(tmp.values)
    gps_rec.append(tmp2.values)

    ax.scatter(tmp,tmp2,marker='^',label=i)
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

hrrr_rec = np.array(hrrr_rec).ravel()
gps_rec = np.array(gps_rec).ravel()

tmp, tmp2 = [],[]
for i in range(len(hrrr_rec)):
    if np.isnan(hrrr_rec[i]) == False and np.isnan(gps_rec[i]) == False:
        tmp.append(hrrr_rec[i])
        tmp2.append(gps_rec[i])

r_stats = linregress(tmp,tmp2)
ax.plot(np.linspace(-30,30,51),r_stats.slope*np.linspace(-30,30,51)+r_stats.intercept,
        color='k')

ax.set_xlim([0,30]);ax.set_ylim([0,30])
ax.text(20,25,'R_value:'+str(round(r_stats.rvalue,3)),fontsize=14)
ax.grid(linestyle=':',linewidth=0.5)
ax.set_ylabel('COSMIC PWV [mm]',fontsize=14);ax.set_xlabel('HRRR PWV [mm]',fontsize=14)
plt.legend()
fig.savefig(OUTDIR+'HRRR_GPS_PWV_scatterplot.png',dpi=1200,
          bbox_inches='tight',facecolor='white', transparent=False)
plt.close()

#=========================================
#  SECT 3. Generate 3-D InSAR PW netCDF
#=========================================
# create InSAR-derived dataset
pwv_insar_abs.attrs["units"] = "millimeter"
pwv_insar_abs.to_netcdf(OUTDIR+'InSAR_derived_PWV.nc')

time.sleep(10)
os.chdir(OUTDIR)
os.system('cdo -setattribute,longitude@units="degrees_east",latitude@units="degrees_north" InSAR_derived_PWV.nc InSAR_derived_PWV_F.nc')
print('OUTFILE: InSAR_derived_PWV_F.nc')
print('Process completed!!')
