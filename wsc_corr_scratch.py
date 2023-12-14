from proj_utils import *
from tidy_data import *
from proj_pods import *
from viz import *
import numpy as np
import datetime as dt
import xarray as xr
import warnings

"""
Get winter correlations
"""

# Load curl anomaly from nc file
pth = '/Users/lillienders/PycharmProjects/gulf-stream/data'
f_wsc = '/wind_stress_curl_anomaly_obs.nc'
ds_wsc = xr.open_dataset(pth+f_wsc)
wsc_anom_array = ds_wsc['wsc anomaly']

# Define a function that returns DECEMBER, and one that returns JANUARY/FEBRUARY, then concatenates the two datasets
def is_dec(month):
    return (month == 12)
wsc_anom_djf = wsc_anom_array.sel(time=is_dec(wsc_anom_array['time.month']))

"""
Get GSI
"""
f = '/Users/lillienders/Desktop/First Generals/altimetry_sla_93_18.nc'
ds = tidy_read(f)
ds['adt'] = (['latitude', 'longitude'], get_contours(f))
ds['sla_std'] = (['latitude', 'longitude'], np.nanmean(ds.sla_std.data, axis=0))

# Calculate GSI using isoline method
gsi_lon, gsi_lat, sla_ts, sla_ts_std = gs_index(ds,ds['adt']*100, alt = True)
gsi_norm = (sla_ts - np.nanmean(sla_ts))/sla_ts_std

a = np.arange(0,300)

ds = ds.sel(longitude=slice(320, 335),latitude = slice(57,65))

#ax.add_patch(Rectangle((-40,57), 15, 8,edgecolor='k',facecolor='none',lw=2))

n_lags = 9
corrs = np.zeros((n_lags,len(ds.latitude),len(ds.longitude)))
for lag in range(n_lags):
    print(lag)
    jan_idx = a[lag::12]
    feb_idx = a[lag+1::12]
    mar_idx = a[lag+2::12]
    djf_idx = np.sort(np.concatenate((jan_idx,feb_idx,mar_idx)))
    djf_gsi = gsi_norm[djf_idx]

    plt.plot(djf_gsi)
    plt.show()

    for lat in range(len(ds.latitude) - 1):
        for lon in range(len(ds.longitude) - 1):
            curl_temp = wsc_anom_djf[:, lat, lon]
            corrs[lag, lat, lon] = np.corrcoef(curl_temp[0:(-(0 + 1))], djf_gsi[0:-1])[0, 1]


    sig = np.zeros((len(ds.latitude), len(ds.longitude)))
    sig[:] = np.nan

    for lat in range(len(ds.latitude)):
        for lon in range(len(ds.longitude)):
            if abs(corrs[lag, lat, lon]) > -0.3 and abs(corrs[lag, lat, lon]) < 0.3:
                sig[lat, lon] = 1
    from matplotlib.patches import Rectangle
    lag_to_plot = 72
    bbox  =[260, 360, 0, 90] # North Atlantic
    crs = ccrs.PlateCarree()
    ax=None
    fig = plt.figure(figsize=(15,10))
    if ax is None:
        ax = plt.subplot(1, 1, 1, projection=crs)
    ax.set_extent(bbox, crs)
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    levels = np.linspace(-0.7,0.7,25)
    colorplot = plt.contourf(ds.longitude, ds.latitude,corrs[lag,:,:], cmap='cmo.balance', shading = 'flat',zorder = 0,levels=levels)
    cbar = plt.colorbar(colorplot,fraction=0.035, pad=0.04)
    cbar.set_label('Correlation Coefficient', size='25', labelpad=25)
    ax.add_patch(Rectangle((-40,57), 15, 8,edgecolor='k',facecolor='none',lw=2))
    ax.add_patch(Rectangle((-45,40), 15, 8,edgecolor='k',facecolor='none',lw=2))
    ax.add_patch(Rectangle((-37,15), 15, 8,edgecolor='k',facecolor='none',lw=2))
    gl.xlabel_style, gl.ylabel_style = {'fontsize': 25}, {'fontsize': 25}

    xorg = ds.longitude.data
    yorg = ds.latitude.data
    nlon = len(xorg)
    nlat = len(yorg)
    xdata = np.reshape(np.tile(xorg, nlat), corrs[lag,:,:].shape)
    ydata = np.reshape(np.repeat(yorg, nlon), corrs[lag, :,:].shape)
    xpoints = xdata[sig == 1]
    ypoints = ydata[sig == 1]
    plt.scatter(xpoints, ypoints, s=0.2, c='k', marker='.', alpha=0.3)
    gl.xlabel_style, gl.ylabel_style = {'fontsize': 20}, {'fontsize': 20}
    plt.gca().autoscale(False)

    plt.show()
print(np.nanmean(abs(corrs),axis=(1,2)))
print("yay!")