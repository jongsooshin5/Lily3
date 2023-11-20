import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from copy import deepcopy

# Get File
f = '/Users/lillienders/Desktop/gulf_stream/Data/b.e11.BRCP85C5CNBDRD.f09_g16.032.pop.h.SSH.200601-208012.nc'

ds = nc.Dataset(f)
time = ds["time"][:]
lat = ds["TLAT"][:]
lon = ds["TLONG"][:]
ssh = ds["SSH"][:]
ssh_av = np.mean(ssh, axis=0)


# Set CRS and Extent
crs = ccrs.PlateCarree()
#bbox  =[270, 320, 20, 50]

# Make Plot
ax=None
fig = plt.figure(figsize=(11, 8.5))
if ax is None:
    ax = plt.subplot(1, 1, 1, projection=crs)
#ax.set_extent(bbox, crs)

# Add Filled Coastline
ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.LAND, facecolor='k', zorder=1)

# Add Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
gl.top_labels = gl.right_labels = False
gl.xformatter = LongitudeFormatter(degree_symbol='')
gl.yformatter = LatitudeFormatter(degree_symbol='')

# Add Colourplot
colorplot = plt.pcolormesh(lon,lat,ssh_av,transform=ccrs.PlateCarree(),
                         #vmin = -100, vmax = 50, zorder=-1,
                          cmap = plt.cm.get_cmap("Spectral_r"), shading ='flat')
plt.title(f"SSH Example Contours")

#cbarax = fig.add_axes([0.9, 0.25, 0.025, 0.5])
cbar = plt.colorbar(colorplot,fraction=0.03, pad=0.04)
cbar.set_label('SSH (cm)', rotation=270, size='12', labelpad=25)

#colorplot = ax.contourf(lon, lat,ssh[500,:,:], 100,transform=ccrs.PlateCarree(),
#                        #vmin=0, vmax=40,
#                        cmap=plt.cm.get_cmap("Spectral_r"), zorder=-1)

#isolines = ax.contour(lon,lat, ssh_av, [-30,-10,10,30],colors='k', linewidths=2)
#plt.clabel(isolines, levels=[-30,-10,10,30], fontsize='10', fmt='%.0f', inline=True,zorder=30)

plt.savefig('full_grid_contours.png', dpi=300)
plt.show()

################
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
"""
import pickle
import scipy.io

from matplotlib.dates import date2num
"""

f = '/Users/lillienders/Desktop/b.e11.BRCP85C5CNBDRD.f09_g16.032.pop.h.SSH.200601-208012.nc'

ds = nc.Dataset(f)
#print(ds)

for var in ds.variables.values():
    print(var)

time = ds["time"][:]
lat = ds["TLAT"][:]
lon = ds["TLONG"][:]
ssh = ds["SSH"][:]
HT = ds["HT"][:]
ssh50 = ssh[50,:,:]
Z = (1 - lat/2 + lat**5 + lon**3) * np.exp(-lat**2 - lon**2)

np.ndarray.view(lat)
#print(time.shape)
#print(lat.shape)
#print(lon.shape)
print(ssh[50,:,:].shape)

#print(lat[0].shape)
#print(lon[1].shape)
#print(infile.__dict__)
#a = infile['title']
#print(a)

#print(type(ssh))
print(type(lon))
print(type(lat))
print(type(Z))
""" 
fig, ax = plt.subplots(1,1,figsize=(12,8))
#plt.plot(lon,lat)
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
colorplot = ax.pcolormesh(lon, lat,ssh[500,:,],
                          vmin = -120, vmax = 50,
                          cmap = plt.cm.get_cmap("Spectral_r"), shading ='auto')
ax.set_xlim(270, 320)
ax.set_ylim(20, 50)
for item in ([ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(25)
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)
#cbarax = fig.add_axes()
cbar = plt.colorbar(colorplot)
cbar.set_label('SSH (cm)', rotation=270, size='25', labelpad=25)
cbar.ax.tick_params(labelsize='25')
#plt.savefig('gulf_stream_ts500.png', dpi=500)
plt.show()
"""
fig, ax = plt.subplots(1,1,figsize=(12,8))
#plt.plot(lon,lat)
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
colorplot = ax.pcolormesh(lon, lat,HT/100,
                          #vmin = -120, vmax = 50,
                          cmap = plt.cm.get_cmap("Spectral_r"), shading ='auto')
ax.set_xlim(270, 320)
ax.set_ylim(20, 50)
for item in ([ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(25)
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(25)
#cbarax = fig.add_axes()
cbar = plt.colorbar(colorplot)
cbar.set_label('SSH (cm)', rotation=270, size='25', labelpad=25)
cbar.ax.tick_params(labelsize='25')
plt.savefig('gulf_stream_ts500.png', dpi=500)
plt.show()

"""
RUN ANALYSIS FOR CESM-LE (UNINITIALIZED)
"""
f_le = '/Users/lillienders/Desktop/cesm_sla_full_ts.nc'
ds_le = xr.open_dataset(f_le) # Read netcdf data into xarray
ds_le = ds_le.sel(year=slice(1985,2005))
ds_le = seasonal_detrend(ds_le)
ds_le = fmt_time(dataset = ds_le)
ds_le = ds_le.sel(longitude=slice(285, 315),latitude=slice(33,46))

# Gulf Stream Index
gsi_lon_le, gsi_lat_le, sla_ts_le, sla_ts_std_le = gs_index(ds_le)

# EOFs for all ensembles
sla = ds_le.to_array(dim='sla_deseasonalized')[-1]
modes = 3
eofs = np.zeros((len(ds_le.Ensemble),modes,len(ds_le.latitude),len(ds_le.longitude)))
pcs = np.zeros((len(ds_le.Ensemble),len(ds_le.time),modes))
per_var = np.zeros((len(ds_le.Ensemble),565))
for ens in range(len(ds_le.Ensemble)):
    sla_ens = np.squeeze(sla[ens,:,:,:])
    eofs[ens,:,:], pcs[ens,:,:], per_var[ens,:] = calc_eofs(sla_ens,num_modes=modes)

# Mean EOFs
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ens_mn_eof = np.nanmean(eofs, axis = 0)
    ens_mn_pcs = np.nanmean(pcs, axis = 0)
    ens_mn_per_var = np.nanmean(per_var, axis = 0)
