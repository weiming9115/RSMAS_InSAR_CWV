#=============================================
#  InSAR Precipitable Water Reconstruction   
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
geo_file = '/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/inputs/geometryRadar.h5'
date_ref = np.loadtxt('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/pic/reference_date.txt')

OUTDIR = '/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/outfiles/'

#=====================================
os.system('rm -rf '+OUTDIR) # remove old folder
os.system('mkdir -p '+OUTDIR) # create OUTDIR 

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
    
    ifg_file = '/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/inputs/ifgramStack.h5'
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

def best_station_Ref(tot_delay_sub, station_id_subdomain, lon_ref, lat_ref, datetime_ref):
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
    data_gps_reloc = tmp.sort_values(by='valid_record',ascending=False)
    
    try:
        return (data_gps_reloc.head(1).index.values[0], 
                data_gps_reloc.head(1).lon.values[0],
                data_gps_reloc.head(1).lat.values[0],data_gps_reloc)
        # (ref_statID, lon_refnew, lat_refnew)
    except:
        return str('None')

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
    cf = ax.contourf(data.longitude,data.latitude,data_showcase_ZTD*100*10)
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
    
    (ref_statID, lon_refnew, lat_refnew, data_gps_reloc) = best_station_Ref(tot_delay_sub
                                                            ,station_id_subdomain, lon_ref, lat_ref, datetime_ref)

    delay_ref_stat = tot_delay_insar.sel(longitude=lon_refnew, latitude=lat_refnew, method='nearest') # 1D timeseries at GPS station
    delay_ref_insar = tot_delay_insar.sel(longitude=lon_ref, latitude=lat_ref, method='nearest') # 1D timeseries at insar default ref.
    offset = delay_ref_insar - delay_ref_stat # offset at each time
    
    # add offset that makes delay at GPS become zero, the new reference point
    tot_delay_insar_reloc = tot_delay_insar + offset 
    
    return ( tot_delay_insar_reloc, ref_statID )

def view_ZTDdiff_relocate(tot_delay_sub_insar, tot_delay_sub, station_id_subdomain, datetime_ref):
    
    fig = plt.figure(figsize=(6,6))
    for i in station_id_subdomain.index.values[:]:
        tmp = tot_delay_sub_insar[i]*100*10 # insar total delay diff [mm]
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
geo_file = '/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/inputs/geometryRadar.h5'
geo = h5py.File(geo_file,'r')
# for key in geo.keys():
#     print(key) #Names of the groups in HDF5 file.
lat = geo['latitude'];
lon = geo['longitude'];
incidence = geo['incidenceAngle'];
axis_bound = [np.min(lat),np.max(lat),np.min(lon),np.max(lon)]; # coordinate bound [South,North,West,East]
axis_bound = [np.unique(lat.value)[1],np.unique(lat.value)[-1],np.unique(lon.value)[0],np.unique(lon.value)[-2]]
print('Box boundaries: ',axis_bound)
date_ref = np.loadtxt('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/pic/reference_date.txt')
print('Reference time: ',date_ref)

# 0. Get InSAR data (processed by rsmas_insar software)
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/InSAR_zenithdisp.nc')
data = data.sel(time=slice(datetime(2014,1,1),datetime(2022,12,31))) # for testing 

# 1. Get GPS data (processed by COSMIC_ProcessDataframe.py) at each acquisition
stat_df = merge_pkl_files(var_name='statInfo',year=[2014,2015,2016,2017,2018,2019,2020,2021,2022])
# remove duplicates in station index (same name but slightly different values in lon lat...)
stat_df = stat_df[~stat_df.index.duplicated(keep='first')]

pwv_df = merge_pkl_files(var_name='pwv',year=[2014,2015,2016,2017,2018,2019,2020,2021,2022])
wet_delay_df = merge_pkl_files(var_name='wet_delay',year=[2014,2015,2016,2017,2018,2019,2020,2021,2022])
dry_delay_df = merge_pkl_files(var_name='final_dry_delay',year=[2014,2015,2016,2017,2018,2019,2020,2021,2022])
pifactor_df = merge_pkl_files(var_name='pifact',year=[2014,2015,2016,2017,2018,2019,2020,2021,2022])

# Data quality control: removing missing values in GPS
pwv_df[pwv_df<0] = np.nan
wet_delay_df[wet_delay_df<0] = np.nan
dry_delay_df[dry_delay_df<0] = np.nan
pifactor_df[pifactor_df<0] = np.nan

# extract t02z data over CONUS stations to match the acquisition time of InSAR
date_list = []
for date in data.time.values[data.time.values < np.datetime64(datetime(2022,12,31,0))]:
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
tot_delay_insar = data.disp_zenith # origional insar data
tot_delay_sub = tot_delay_acqu[station_id_subdomain.index] # gps data

hrUTC = str(tot_delay_sub.index[0].hour).zfill(2)
datetime_ref = datetime.strptime(str(date_ref)[:-2]+hrUTC,'%Y%m%d%H')
(lon_ref, lat_ref) = find_latlon_ref(lon,lat)

(ref_statID, lon_refnew, lat_refnew, data_gps_reloc) = best_station_Ref(tot_delay_sub
                                                            ,station_id_subdomain, lon_ref, lat_ref, datetime_ref)
(tot_delay_insar_reloc, ref_statID) = reset_tot_delay_ref(tot_delay_insar, tot_delay_sub, lon_ref, lat_ref
                                            , station_id_subdomain, datetime_ref) # relocate ref
data_gps_reloc.to_pickle(OUTDIR+'data_GPSinfo_BestRef.pkl')
print('OUTFILE: data_GPSinfo_BestRef.pkl')
#--------------------------------------------------------------------------
# selecting subdomain dataframes
tot_delay_sub = tot_delay_acqu[station_id_subdomain.index] # gps 
pwv_sub = pwv_acqu[station_id_subdomain.index]             # gps
pifactor_sub = pifactor_acqu[station_id_subdomain.index]   # gps

tot_delay_sub_insar = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = tot_delay_insar_reloc.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'disp_zenith': i})
    tot_delay_sub_insar = pd.concat([tot_delay_sub_insar,tmp],axis=1)

# 2. Adding back the offset removed in ZTD interferograms by GPS 
hrUTC = str(tot_delay_sub.index[0].hour).zfill(2)
datetime_ref = datetime.strptime(str(date_ref)[:-2]+hrUTC,'%Y%m%d%H')
print('Reference GPS station: ', ref_statID)
print('Reference datetime: ', datetime_ref)

fig = view_domain(tot_delay_insar_reloc[6,:,:],station_id_subdomain, lon_refnew, lat_refnew)
fig.savefig(OUTDIR+'InSAR_GPS_view.png',bbox_inches='tight',transparent=False)
print('OUTFILE: InSAR_GPS_view.png')

fig = view_ZTDdiff_relocate(tot_delay_sub_insar, tot_delay_sub, station_id_subdomain, datetime_ref)
fig.savefig(OUTDIR+'InSAR_ZTDdiff_relocate.png',bbox_inches='tight',transparent=False)
print('OUTFILE: InSAR_ZTDdiff_relocate.png')

# =========================================================
#  SECT. B start InSAR-CWV retrival if ref_statID exists!!!!!!!!!!!
#  some swathes with ref point over sea... or no GPS nearby
# =========================================================

# the reference point on InSAR map 
S1_file = glob('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/S1_*.he5')[0]
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
    sar_ps_diff[t,:,:] = remap_hrrr2SAR(ps_diff[t,:,:]) # regrid pressure diff into sar grid

sar_dry_delay = hydro_factor*sar_ps_diff #[m/pa]*[pa]
sar_dry_delay = xr.DataArray(sar_dry_delay,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='dry_delay')
sar_dry_delay = sar_dry_delay.to_dataset(name='dry_delay')

# wet delay
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_pwat.nc')
pwat_acqu = data.pwat
pwat_diff = pwat_acqu-pwat_acqu.sel(time=datetime_ref)
sar_pwat_diff = np.zeros((len(date_list),len(lat_sar),len(lon_sar)))

for t in range(sar_pwat_diff.shape[0]):
    sar_pwat_diff[t,:,:] = remap_hrrr2SAR(pwat_diff[t,:,:]) # regrid pressure diff into sar grid

sar_wet_delay = 6.5*sar_pwat_diff #[m/pa]*[pa]
sar_wet_delay = xr.DataArray(sar_wet_delay,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='dry_delay')
sar_wet_delay = sar_wet_delay.to_dataset(name='dry_delay')

tot_delay_acqu = wet_delay_acqu + dry_delay_acqu
tot_delay_GPS = tot_delay_acqu[ref_statID] # GSP total delay at the reference point
# extract tot_delay at the new reference point from HRRR
tot_delay_hrrr = sar_dry_delay + sar_wet_delay
tot_delay_hrrr = tot_delay_hrrr.sel(longitude=lon_refnew,latitude=lat_refnew,method='nearest').dry_delay



dry_delay_sub = dry_delay_acqu[station_id_subdomain.index]
dry_delay_GPS = dry_delay_acqu[ref_statID] # GSP dry delay at the reference point

######### remove offset according to GPS/HRRR (close to the reference point) ############
offset = (tot_delay_GPS-tot_delay_sub.loc[datetime_ref][ref_statID]) - tot_delay_sub_insar[ref_statID]*10*100 # [mm
### if there's no GPS signals (nearly half of time)
offset_byhrrr = (tot_delay_hrrr-tot_delay_hrrr.sel(time=datetime_ref)).values - tot_delay_sub_insar[ref_statID].values*10*100
offset_byhrrr = pd.Series(offset_byhrrr, index=offset.index, name=offset.name)

valid_record = len(np.where(offset.values>0)[0])
offset_best = np.empty(len(offset))
for n,val in enumerate(offset):
    if np.isnan(val) == 1:
        offset_best[n] = offset_byhrrr[n] 
    else:
        offset_best[n] = offset[n]
offset_best = pd.Series(offset_best, index=offset.index, name=offset.name)

fig = plot_ZTDoffset_series(offset_best)
fig.savefig(OUTDIR+'InSAR_ZTDoffset_best.png',dpi=600,bbox_inches='tight',transparent=False)
print('OUTFILE: InSAR_ZTDoffset_best.png') 
### note: in this way, the ZTD of InSAR should be identical to GPS or HRRR (if GPS not valid) #####

# plotting corrected ZTD 
fig = plt.figure(figsize=(5,5))
insar_rec = []; gps_rec = []

for i in station_id_subdomain.index.values[:]:
    tmp = tot_delay_sub_insar[i]*100*10 # insar total delay diff [mm]
    tmp2 = tot_delay_sub[i]-tot_delay_sub.loc[datetime_ref][i]-offset_best # COSMIC total delay diff
        
    insar_rec.append(tmp.values)
    gps_rec.append(tmp2.values)
    
    plt.scatter(tmp,tmp2,marker='^',label=i)
plt.legend()
plt.ylabel('COSMIC ZTD diff. [mm]',fontsize=13)
plt.xlabel('InSAR ZTD diff. [mm]',fontsize=13)
fig.savefig(OUTDIR+'InSAR_ZTDdiff_relocate.png')

insar_rec = np.array(insar_rec).ravel()
gps_rec = np.array(gps_rec).ravel()

tmp, tmp2 = [],[]
for i in range(len(insar_rec)):
    if np.isnan(insar_rec[i]) == False and np.isnan(gps_rec[i]) == False:
        tmp.append(insar_rec[i])
        tmp2.append(gps_rec[i])

r_stats = linregress(tmp,tmp2)
plt.plot(np.linspace(-45,45,51),r_stats.slope*np.linspace(-45,45,51)+r_stats.intercept,
        color='k')
plt.plot(np.linspace(-45,45,51),1.0*np.linspace(-45,45,51),color='lightgrey')
plt.xlim([-45,45]);plt.ylim([-45,45])
plt.text(10,30,'R_value:'+str(round(r_stats.rvalue,3)),fontsize=14)
plt.grid(linestyle=':',linewidth=0.5)
plt.ylabel('COSMIC ZTD [mm]');plt.xlabel('InSAR ZTD [mm]')
plt.legend()
fig.savefig(OUTDIR+'ZTD_addoffset_scatterplot.png',dpi=600,
           bbox_inches='tight',facecolor='white', transparent=False)
plt.close()

#==============================================
# 3.ADD HRRR to (1) generate ZTD for correction when reference GPS has missing data
#               (2) provide dry delay (ZDD) at each InSAR pixel to derive wet delay (ZWD)
#================================================
# the reference point on InSAR map, near GPS
S1_file = glob('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/S1_*.he5')[0]
data_ifgrams = h5py.File(S1_file, 'r')
geo_info = data_ifgrams['HDFEOS']['GRIDS']['timeseries']['geometry']
lon_ifgrams = geo_info['longitude']
lat_ifgrams = geo_info['latitude']

lon_sar = np.linspace(axis_bound[2],axis_bound[3],lon_ifgrams.shape[1])
lat_sar = np.linspace(axis_bound[0],axis_bound[1],lon_ifgrams.shape[0])

# generate the 3-D dry delay field from HRRR surface pressure outputs
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_ps.nc')
ps_acqu = data.sp
ps_diff = ps_acqu-ps_acqu.sel(time=datetime_ref)
hydro_factor = 1e-6*0.776*287.15/9.8 # [m/pa]
sar_ps_diff = np.zeros((len(date_list),len(lat_sar),len(lon_sar)))

for t in range(sar_ps_diff.shape[0]):
    sar_ps_diff[t,:,:] = remap_hrrr2SAR(ps_diff[t,:,:]) # regrid pressure diff into sar grid

sar_dry_delay = hydro_factor*sar_ps_diff #[m/pa]*[pa]
sar_dry_delay = xr.DataArray(sar_dry_delay,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='dry_delay')
sar_dry_delay = sar_dry_delay.to_dataset(name='dry_delay')

# ================================= 
#    4. Derive InSAR wet delay 
# =================================
# estimate the zenith wet delay by combining InSAR and atmospheric vars from HRRR 
wet_delay_diff_insar = 100*10*(tot_delay_insar_reloc - sar_dry_delay) # [mm]
wet_delay_diff_insar = wet_delay_diff_insar.rename({'dry_delay':'wet_delay'}) # rename to the correct name
# station-wise comparison between GPS PWV
wet_delay_sub = wet_delay_acqu[station_id_subdomain.index]
wet_delay_sub = wet_delay_sub.replace(-99.900002,np.nan)

wet_delay_diff_sub_insar = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    # find nearest point close to GPS stations
    tmp = wet_delay_diff_insar.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'wet_delay': i})
    wet_delay_diff_sub_insar = pd.concat([wet_delay_diff_sub_insar,tmp],axis=1)

dry_delay_diff_sub_insar = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    # find nearest point close to GPS stations
    tmp = sar_dry_delay.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'dry_delay': i})
    dry_delay_diff_sub_insar = pd.concat([dry_delay_diff_sub_insar,tmp],axis=1)

# adding HRRR ZWD_diff to compensate for missing GPS times at the refenrence point
pwv_hrrr = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_pwat.nc')
pwv_hrrr = pwv_hrrr.sel(time=slice(datetime(2014,1,1),datetime(2022,12,31)))
pwv_diff_hrrr = pwv_hrrr - pwv_hrrr.sel(time=datetime_ref) # pwv_diff to converted into wet_delay_diff_hrrr

# use 6.5 as the conversion factor to build wet_delay from HRRR
wet_delay_diff_hrrr = np.copy(tot_delay_insar_reloc)*np.nan

for t in range(len(date_list)):
    try:
        wet_delay_diff_hrrr[t,:,:] = remap_hrrr2SAR(pwv_diff_hrrr.pwat[t,:,:]*6.5)
    except:
        continue
wet_delay_diff_hrrr = xr.DataArray(wet_delay_diff_hrrr,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='wet_delay')
wet_delay_diff_hrrr = wet_delay_diff_hrrr.to_dataset(name='wet_delay')

# adding local offset back to InSAR data based on GPS 

# there is a difference between sar_dry_delay form HRRR and gps_dry_delay at reference
# wet_delay_insar should be adjusted to wet_delay_gps, like what we have in total delay.
sar_dry_delay_sub = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = sar_dry_delay.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'dry_delay': i})
    sar_dry_delay_sub = pd.concat([sar_dry_delay_sub,tmp],axis=1)    
offset_dry = (dry_delay_GPS-dry_delay_sub.loc[datetime_ref][ref_statID]) - sar_dry_delay_sub[ref_statID]*10*100

######
sar_dry_delay = sar_dry_delay.sel(longitude=lon_refnew,latitude=lat_refnew,method='nearest').dry_delay
offset_dry_byhrrr = (sar_dry_delay-sar_dry_delay.sel(time=datetime_ref)).values -  sar_dry_delay_sub[ref_statID].values*10*100
offset_dry_byhrrr = pd.Series(offset_dry_byhrrr, index=offset_dry.index, name=offset_dry.name)

###### offset_dry best - replacing missing GPS by HRRR #####
offset_dry_best = np.empty(len(offset_dry))
for n,val in enumerate(offset_dry):
    if np.isnan(val) == 1:
        offset_dry_best[n] = offset_dry_byhrrr[n] 
    else:
        offset_dry_best[n] = offset_dry[n]
offset_dry_best = pd.Series(offset_dry_best, index=offset_dry.index, name=offset_dry.name)

#################  Last step for reconstruction ZWD : adding local offset back to InSAR data based on GPS/HRRR ######
wet_delay_diff_insar_revised = np.zeros(wet_delay_diff_insar.wet_delay.shape)

for i,t in enumerate(date_list):
    wet_delay_diff_insar_revised[i,:,:] = wet_delay_diff_insar.wet_delay.sel(time=t) + (offset_best-offset_dry_best).loc[t]
wet_delay_diff_insar_revised = xr.DataArray(wet_delay_diff_insar_revised,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='wet_delay') 
wet_delay_diff_insar_revised = wet_delay_diff_insar_revised.to_dataset(name='wet_delay')

wet_delay_diff_sub_insar_re = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = wet_delay_diff_insar_revised.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'wet_delay': i})
    wet_delay_diff_sub_insar_re = pd.concat([wet_delay_diff_sub_insar_re,tmp],axis=1)

wet_delay_diff_sub_hrrr = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = wet_delay_diff_hrrr.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'wet_delay': i})
    wet_delay_diff_sub_hrrr = pd.concat([wet_delay_diff_sub_hrrr,tmp],axis=1)

# remove offset to get relative ZWD reference to GPS and divided by pi_factor (6.5 here)
# to convert ZWD into PWV at the reference time and point
# PWV(x,t) = Pi*(ZDT(x,t) + ZDT_gps(x_ref,t)) + PWV(x,tref)
pwv_sub_hrrr = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = pwv_hrrr.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'pwat': i})
    pwv_sub_hrrr = pd.concat([pwv_sub_hrrr,tmp],axis=1)

pwv_insar_test = wet_delay_diff_insar_revised/6.5 
pwv_sub_insar_test = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = pwv_insar_test.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'wet_delay': i})
    # add pwv reference back to ZWD
    if np.isnan(pwv_sub.loc[datetime_ref][i]) == False: # if GPS nan
        tmp2 = pwv_sub.loc[datetime_ref][i]
    else: # using HRRR instead, making sure that we have a complete InSAR-derived PWV dataset if GPS fails
        tmp2 = pwv_sub_hrrr.loc[datetime_ref][i]
    pwv_sub_insar_test = pd.concat([pwv_sub_insar_test,tmp.add(tmp2)],axis=1)


# ZWD adjusted timeseries of InSAR and selected GPS stations -- see how well InSAR matches GPS
station_avail = data_gps_reloc.index.values[:6] # pick 6 stations, including the reference GPS
nrow = [0,0,1,1,2,2]
ncol = [0,1,0,1,0,1]

fig,ax = plt.subplots(3,2,figsize=(18,7))

for i,stat in enumerate(station_avail):
    
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,wet_delay_diff_sub_insar_re[stat], 'or') # original InSAR ZWD diff
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,wet_delay_diff_sub_hrrr[stat],'grey',linewidth=1) # original HRRR ZWD diff
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,wet_delay_sub[stat]-wet_delay_sub.loc[datetime_ref][stat],'ob')
    loc_str = ('  ('+str(round(station_id_subdomain.loc[stat]['lon'],2)) 
               +', '+str(round(station_id_subdomain.loc[stat]['lat'],2)) + ')')
    ax[nrow[i],ncol[i]].set_title(stat + loc_str,fontsize=13, fontweight='bold')
    ax[nrow[i],ncol[i]].set_ylabel('ZWD_diff [mm]',fontsize=13)
    ax[nrow[i],ncol[i]].set_ylim([-40,120])

    ax[nrow[i],ncol[i]].grid(linestyle=':',linewidth=0.5)
    ax[nrow[i],ncol[i]].legend(['InSAR_re','HRRR','COSMIC GPS'],loc=2)
    ax[nrow[i],ncol[i]].plot(datetime_ref,0,color='gold',marker='s')
    ax[nrow[i],ncol[i]].plot(pwv_sub.index,np.zeros(len(pwv_sub.index)),
                             color='lightgrey')
    ax[nrow[i],ncol[i]].set_xlim([date_list[0],date_list[-1]])
    ax[nrow[i],ncol[i]].spines['right'].set_visible(False)
    ax[nrow[i],ncol[i]].spines['top'].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR+'ZWD_diff_timeseries.png',dpi=600,
           bbox_inches='tight',facecolor='white', transparent=False)

#=============================================
#  5. Scatterplot of InSAR  PW at GPS staitons
#==============================================
fig,ax = plt.subplots(1,1,figsize=(5,5))

insar_rec = []; gps_rec = []; 

for i in station_id_subdomain.index.values[:]:
    tmp = pwv_sub_insar_test[i] # insar derived PWV
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
fig.savefig(OUTDIR+'InSAR_GPS_PWV_scatterplot.png',dpi=600,
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
fig.savefig(OUTDIR+'HRRR_GPS_PWV_scatterplot.png',dpi=600,
          bbox_inches='tight',facecolor='white', transparent=False)

#=========================================
#  SECT 3. Generate 3-D InSAR PW netCDF  
#=========================================
# create InSAR-derived dataset
pwv_insar_abs = np.copy(wet_delay_diff_insar.wet_delay)
pwv_hrrr_ref = pwv_hrrr.sel(time=datetime_ref).pwat
pwv_hrrr_ref_remap = remap_hrrr2SAR(pwv_hrrr_ref)

for i,t in enumerate(date_list):
    # PW_insar[t] = ZWD_insar[t]/6.5 + PW[t_ref]
    pwv_insar_abs[i,:,:] = wet_delay_diff_insar_revised.wet_delay[i,:,:]/6.5 + pwv_hrrr_ref_remap[:,:]
pwv_insar_abs = xr.DataArray(pwv_insar_abs,dims=('time','latitude','longitude')
                            ,coords=(date_list,lat_sar,lon_sar),name='pwat')
pwv_insar_abs.to_dataset(name='pwat')
pwv_insar_abs.attrs["units"] = "millimeter"
pwv_insar_abs.to_netcdf(OUTDIR+'InSAR_derived_PWV.nc')

time.sleep(10)
os.chdir(OUTDIR)
os.system('cdo -setattribute,longitude@units="degrees_east",latitude@units="degrees_north" InSAR_derived_PWV.nc InSAR_derived_PWV_F.nc')
print('OUTFILE: InSAR_derived_PWV_F.nc')
print('Process completed!!')
