import netCDF4 as nc
import numpy as np
import xarray as xr
import warnings
from eofs.xarray import Eof
import matplotlib.pyplot as plt
import statsmodels.api as sm


def gs_index_joyce(dataset):
    """
    Calculate the Locations of Gulf Stream Indices using Terry Joyce's Maximum Standard Deviation Method (Pérez-Hernández and Joyce (2014))
    Inputs:
    - dataset: containing longitude, latitude, sla, sla_std
    Returns:
    - gsi_lon: longitudes of gulf stream index points
    - gsi_lat: latitudes of gulf stream index points
    - std_ts: time series of gulf stream index
    """
    # Load data array, trim longtiude to correct window
    ds = dataset.sel(longitude=slice(290, 308))
    # Coarsen data array to nominal 1 degree (in longitude coordinate)
    crs_factor = int(1 / (ds.longitude.data[1] - ds.longitude.data[
        0]))  # Calculate factor needed to coarsen by based on lon. resolution
    if crs_factor != 0:
        ds = ds.coarsen(longitude=crs_factor,
                        boundary='pad').mean()  # Coarsen grid in longitude coord using xarray built in
    gsi_lon = ds.longitude.data  # Save gsi longitudes in array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Take time average to produce a single field
        mn_std = ds.sla_std.data
        mn_sla = np.nanmean(ds.sla.data, axis=0)
        # Calculate location (latitude) of maximum standard deviation
        gsi_lat_idx = np.nanargmax(mn_std, axis=0)
        gsi_lat = ds.latitude[gsi_lat_idx].data
        sla_flt = ds.sla.data.reshape(len(ds.time), len(ds.latitude), len(ds.longitude))
        temp = np.zeros(len(ds.longitude))
        sla_ts = np.zeros(len(ds.time))
        sla_ts_std = np.zeros(len(ds.time))

        for t in range((len(ds.time))):
            for lon in range(len(ds.longitude)):
                temp[lon] = sla_flt[t, gsi_lat_idx[lon], lon]
            sla_ts[t] = np.nanmean(temp)
            sla_ts_std[t] = np.nanstd(temp)
    return (gsi_lon, gsi_lat, sla_ts, sla_ts_std)


def gs_index(dataset, adt, alt = False):
    # Get Gulf Stream Index Locations (Contour w Max Standard Deviation)
    ds = dataset
    if alt == True:
        x, y = get_contour_info(ds.longitude, ds.latitude, abs(adt),
                                contour_to_get=get_max_contour(ds, adt, contours_to_try = np.linspace(30, 50, 21)))
    else:
        x, y = get_contour_info(ds.longitude, ds.latitude, abs(adt), contour_to_get=get_max_contour(ds, adt))
    subset_ind = []
    for k in np.linspace(-70, -55, 16):
        subset_ind.append(np.nanargmin(abs((x - 360) - k)))
    gsi_lon, gsi_lat = x[subset_ind], y[subset_ind]

    sla_ts = np.zeros(len(ds.time))
    sla_ts_std = np.zeros(len(ds.time))
    temp = np.zeros(len(gsi_lon))

    for t in range((len(ds.time))):
        for x in range(len(temp)):
            temp[x] = ds['sla'][
                t, np.nanargmin(abs(ds.latitude.data - gsi_lat[x])), np.nanargmin(abs(ds.longitude.data - gsi_lon[x]))]
        sla_ts[t] = np.nanmean(temp)
        sla_ts_std[t] = np.nanstd(temp)

    return (gsi_lon, gsi_lat, sla_ts, sla_ts_std)


def get_max_contour(dataset, adt, contours_to_try = np.linspace(5, 30, 26)):
    ds = dataset.sel(longitude=slice(290, 308), latitude=slice(36, 45))
    std_field = np.nanstd(ds.sla, axis=0)
    std_contours = np.zeros(len(contours_to_try))
    for c in range(len(contours_to_try)):
        x_temp, y_temp = get_contour_info(ds.longitude, ds.latitude, abs(adt), contour_to_get=int(contours_to_try[c]))
        temp = 0
        for x in range(len(x_temp)):
            temp = temp + std_field[
                np.nanargmin(abs(ds.latitude.data - y_temp[x])), np.nanargmin(abs(ds.longitude.data - x_temp[x]))]
        std_contours[c] = temp/len(x_temp)
    contour_to_use = int(contours_to_try[np.nanargmax(std_contours)])
    return (contour_to_use)


def get_contour_info(x_data, y_data, bthy_data, contour_to_get=40):
    bbox = [280, 308, 33, 46]  # bind contours to GS region

    lon_min = np.nanargmin(abs(x_data - bbox[0]))
    lon_max = np.nanargmin(abs(x_data - bbox[1]))
    lat_min = np.nanargmin(abs(y_data - bbox[2]))
    lat_max = np.nanargmin(abs(y_data - bbox[3]))

    contours = plt.contour(x_data[lon_min:lon_max], y_data[lat_min:lat_max],
                           abs(bthy_data[lat_min:lat_max, lon_min:lon_max]),
                           levels=[contour_to_get], colors='k', zorder=10, linewidths=0.75)
    x, y = [], []
    for item in contours.collections:
        for i in item.get_paths():
            v = i.vertices
            x_temp = (v[:, 0])
            y_temp = (v[:, 1])
            if len(x_temp) > len(x):
                x = x_temp
                y = y_temp
    for i in range(0, len(x)):
        for j in range(i + 1, len(x)):
            if (x[i] - 0.8 < x[j] < x[i] + 0.8):
                x[j] = np.nan
                y[j] = np.nan
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    plt.close()
    return (x, y)

def smooth_data(t_series, window=12):
    t_series_smoothed = np.zeros(int(len(t_series) / window))
    for t in range(int(len(t_series) / window)):
        t_series_smoothed[t] = np.nanmean(t_series[t * window:t * window + window])
    return (t_series_smoothed)


def get_acf(t_series):
    acf = sm.tsa.stattools.acf(t_series, adjusted=False, nlags=59)
    tau = int(np.squeeze(np.argwhere(np.diff(np.sign(acf)))[0]))

    n_eff = np.zeros(len(t_series))
    for t in range(len(t_series)):
        n_eff[t] = (len(t_series) - t) / tau
    return (acf, n_eff)


def calc_eofs(array, num_modes=1):
    if 'latitude' in array.dims:
        coslat = np.cos(np.deg2rad(array.coords['latitude'].values))
        wgts = np.sqrt(coslat)[..., np.newaxis]
        solver = Eof(array, weights=wgts)
    else:
        solver = Eof(array)
    eofs = np.squeeze(solver.eofs(neofs=num_modes))
    pcs = np.squeeze(solver.pcs(npcs=num_modes, pcscaling=1))
    per_var = solver.varianceFraction()
    return (eofs, pcs, per_var)