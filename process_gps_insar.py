#=============================================
#  InSAR Precipitable Water Reconstruction   
#=============================================
import os
import sys
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

def nearest_station_Ref(station_id_subdomain, lon_ifgrams, lat_ifgrams):

    ## string searching test
    string1 = 'REF_X'
    string2 = 'REF_Y'
    file = open('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/inputs/ifgramStack.out', "r")

    for line in file:
        # checking condition for string found or not
        if string1 in line:
            REF_X = int(line[10:]) # capture reference position, X
            break

    for line in file:
        # checking condition for string found or not
        if string2 in line:
            REF_Y = int(line[10:]) # capture reference position, X
            break
    file.close()

    lon_ref = lon[REF_Y,REF_X] # lat of reference point 
    lat_ref = lat[REF_Y,REF_X] # lon of reference point
    
    station_id_subdomain['dist2ref'] = np.sqrt((station_id_subdomain['lon'] - lon_ref)**2 + 
                                           (station_id_subdomain['lat'] - lat_ref)**2)
    return station_id_subdomain.sort_values(by='dist2ref').head(1).index.values[0]

#=============== END FUNCTIONS ===============

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
data = data.sel(time=slice(datetime(2014,1,1),datetime(2019,12,31))) # for testing 

# 1. Get GPS data (processed by COSMIC_ProcessDataframe.py) at each acquisition
stat_df = merge_pkl_files(var_name='statInfo',year=[2014,2015,2016,2017,2018,2019])
# remove duplicates in station index (same name but slightly different values in lon lat...)
stat_df = stat_df[~stat_df.index.duplicated(keep='first')]

pwv_df = merge_pkl_files(var_name='pwv',year=[2014,2015,2016,2017,2018,2019])
wet_delay_df = merge_pkl_files(var_name='wet_delay',year=[2014,2015,2016,2017,2018,2019])
dry_delay_df = merge_pkl_files(var_name='final_dry_delay',year=[2014,2015,2016,2017,2018,2019])
pifactor_df = merge_pkl_files(var_name='pifact',year=[2014,2015,2016,2017,2018,2019])

# Data quality control: removing missing values in GPS
pwv_df[pwv_df<0] = np.nan
wet_delay_df[wet_delay_df<0] = np.nan
dry_delay_df[dry_delay_df<0] = np.nan
pifactor_df[pifactor_df<0] = np.nan

# extract t02z data over CONUS stations to match the acquisition time of InSAR
date_list = []
for date in data.time.values[data.time.values < np.datetime64(datetime(2019,12,31,0))]:
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

# station-wise comparison between GPS and InSAR zenith delay
tot_delay_sub = tot_delay_acqu[station_id_subdomain.index]
tot_delay_insar = data.disp_zenith
pwv_sub = pwv_acqu[station_id_subdomain.index]
pifactor_sub = pifactor_acqu[station_id_subdomain.index]

tot_delay_sub_insar = pd.DataFrame()
for i in station_id_subdomain.index.values:
        
    lon_sel = station_id_subdomain.loc[i]['lon']
    lat_sel = station_id_subdomain.loc[i]['lat']
    tmp = tot_delay_insar.sel(longitude=lon_sel,latitude=lat_sel,method='nearest')
    tmp = tmp.to_dataframe().drop(columns=['latitude','longitude']).rename(columns={'disp_zenith': i})
    tot_delay_sub_insar = pd.concat([tot_delay_sub_insar,tmp],axis=1)

# 2. Adding back the offset removed in ZTD interferograms by GPS 
tot_delay_acqu = wet_delay_acqu + dry_delay_acqu
# return reference gps station ID for later adjustment
ref_statID = nearest_station_Ref(station_id_subdomain, lon, lat)
hrUTC = str(tot_delay_sub.index[0].hour).zfill(2)
datetime_ref = datetime.strptime(str(date_ref)[:-2]+hrUTC,'%Y%m%d%H')
print('Reference GPS station: ', ref_statID)
print('Reference datetime: ', datetime_ref)

tot_delay_TONO = tot_delay_acqu[ref_statID] # GSP total delay at the reference point
dry_delay_sub = dry_delay_acqu[station_id_subdomain.index]
dry_delay_TONO = dry_delay_acqu[ref_statID] # GSP dry delay at the reference point

# remove offset according to TONO (close to the reference point)
offset = (tot_delay_TONO-tot_delay_sub.loc[datetime_ref][ref_statID]) - tot_delay_sub_insar[ref_statID]*10*100 # [mm]

# plotting corrected ZTD 
fig = plt.figure(figsize=(5,5))
insar_rec = []; gps_rec = []

for i in station_id_subdomain.index.values[:]:
    tmp = tot_delay_sub_insar[i]*100*10 # insar total delay diff [mm]
    tmp2 = tot_delay_sub[i]-tot_delay_sub.loc[datetime_ref][i]-offset # COSMIC total delay diff
        
    insar_rec.append(tmp.values)
    gps_rec.append(tmp2.values)
    
    plt.scatter(tmp,tmp2,marker='^',label=i)
plt.legend()
plt.ylabel('COSMIC zenith delay [mm]',fontsize=13)
plt.xlabel('InSAR zenith delay [mm]',fontsize=13)

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
fig.savefig(OUTDIR+'ZTD_addoffset_scatterplot.pdf',dpi=200,
           bbox_inches='tight',facecolor='white', transparent=False)
plt.close()

# 3.ADD HRRR to (1) generate ZTD for correction when reference GPS has missing data
#               (2) provide dry delay (ZDD) at each InSAR pixel to derive wet delay (ZWD)
#Pi_factor = xr.open_dataset('/data2/willytsai/InSAR_HRRR/HRRR_Pi_factor_NEVADA.nc')

# the reference point on InSAR map, near TONO
data_ifgrams = h5py.File('/data2/willytsai/InSAR_HRRR/'+CASE_ID+'/mintpy/S1_IW123_166_0121_0140_20150322_XXXXXXXX.he5', 'r')
geo_info = data_ifgrams['HDFEOS']['GRIDS']['timeseries']['geometry']
lon_ifgrams = geo_info['longitude']
lat_ifgrams = geo_info['latitude']

lon_sar = np.linspace(axis_bound[2],axis_bound[3],lon_ifgrams.shape[1])
lat_sar = np.linspace(axis_bound[0],axis_bound[1],lon_ifgrams.shape[0])

# generate the 3-D dry delay field from HRRR surface pressure outputs
data = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_ps.nc')
ps_acqu = data.pressure
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
wet_delay_diff_insar = 100*10*(tot_delay_insar - sar_dry_delay) # [mm]
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
#pwv_diff_hrrr = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_PW_diff_NEVADA.nc')
#pwv_diff_hrrr = pwv_diff_hrrr.sel(time=slice(datetime(2014,1,1),datetime(2019,12,31)))
pwv_hrrr = xr.open_dataset('/data2/willytsai/InSAR_HRRR/auto_framework/'+CASE_ID+'/HRRR_regrid3km_pwat.nc')
pwv_hrrr = pwv_hrrr.sel(time=slice(datetime(2014,1,1),datetime(2019,12,31)))
pwv_diff_hrrr = pwv_hrrr - pwv_hrrr.sel(time=datetime_ref) # pwv_diff to converted into wet_delay_diff_hrrr

# use 6.5 as the conversion factor to build wet_delay from HRRR
wet_delay_diff_hrrr = np.copy(tot_delay_insar)*np.nan

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
offset_dry = (dry_delay_TONO-dry_delay_sub.loc[datetime_ref][ref_statID]) - sar_dry_delay_sub[ref_statID]*10*100

wet_delay_diff_insar_revised = np.zeros(wet_delay_diff_insar.wet_delay.shape)

for i,t in enumerate(date_list):
    wet_delay_diff_insar_revised[i,:,:] = wet_delay_diff_insar.wet_delay.sel(time=t) + (offset-offset_dry).loc[t]
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

# remove offset to get relative ZWD reference to TONO and divided by pi_factor (6.5 here)
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
fig.savefig(OUTDIR+'InSAR_GPS_PWV_scatterplot.pdf',dpi=200,
          bbox_inches='tight',facecolor='white', transparent=False)
plt.close()

#=========================================
#  6. Generate 3-D InSAR PW netCDF  
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
os.chdir(OUTDIR)
os.system('cdo -setattribute,longitude@units="degrees_east",latitude@units="degrees_north" InSAR_derived_PWV.nc InSAR_derived_PWV_F.nc')
print('OUTFILE: InSAR_derived_PWV_F.nc')
print('Process completed!!')
