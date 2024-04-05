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
        x, y, std  = get_contour_info(ds,contour_to_get=[33])
    else:
        x, y, std = get_contour_info(ds, contour_to_get=get_max_contour(ds, adt))
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
    contour_to_get = get_max_contour(ds, adt, contours_to_try=np.linspace(30, 50, 21))
    return (gsi_lon, gsi_lat, sla_ts, sla_ts_std)


def get_max_contour(dataset, adt, contours_to_try = np.linspace(5, 30, 26)):
    ds = dataset.sel(longitude=slice(289, 308), latitude=slice(36, 45))
    std_contours = np.zeros(len(contours_to_try))
    for c in range(len(contours_to_try)):
        x_temp, y_temp, std_contours[c] = get_contour_info(ds, contour_to_get=int(contours_to_try[c]))
    contour_to_use = int(contours_to_try[np.nanargmax(std_contours)])
    return (contour_to_use)

def get_contour_info(ds, contour_to_get=40):
    ds = ds.sel(longitude=slice(289, 308), latitude=slice(36, 45))
    x_data = ds.longitude
    y_data = ds.latitude
    std_field = np.nanstd(ds.sla, axis=0)

    contours = plt.contour(x_data, y_data,
                           abs(ds.adt),
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
            temp = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for lon in range(len(x)):
                    temp = temp + std_field[
                        np.nanargmin(abs(ds.latitude.data - y[lon])), np.nanargmin(abs(ds.longitude.data - x[lon]))]
                std_contour = temp/len(x)
    for i in range(0, len(x)):
        for j in range(i + 1, len(x)):
            if (x[i] - 0.1 < x[j] < x[i] + 0.1):
                if y[j] > y[i]:
                    x[i] = np.nan
                    y[i] = np.nan
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    x_lon = np.array([289.875, 290.875, 291.875, 292.875, 293.875, 294.875, 295.875, 296.875, 297.875,
         298.875, 299.875, 300.875, 301.875, 302.875,303.875, 304.875])
    check_x_ord = x[0] > x[-1]
    if x[0] > x[-1]:
        x = np.flip(x)
        y = np.flip(y)
    y_lat = np.interp(x_lon,x,y)
    plt.close()
    return (x_lon, y_lat, std_contour)

def smooth_data(t_series, window=12):
    t_series_smoothed = np.zeros(int(len(t_series) / window))
    for t in range(int(len(t_series) / window)):
        t_series_smoothed[t] = np.nanmean(t_series[t * window:t * window + window])
    return (t_series_smoothed)


def get_acf(t_series):
    acf = sm.tsa.stattools.acf(t_series, adjusted=False, nlags=59)
    if len(np.argwhere(np.diff(np.sign(acf)))) == 0:
        tau = len(acf)
    else:
        tau = int(np.squeeze(np.argwhere(np.diff(np.sign(acf)))[0]))+1

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

def lagged_corrs(var_one,var_two,nlags):
    """
        Calculate lagged correlation coeffients (Pearson Correlation)
        Inputs:
        - var_one: first variable in correlation [time x lat x longitude] of type xarray
        - var_two: second variable in correlation [time x lat x longitude] of type xarray (must have same dimensions as var_one)
        - nlags: number of lags to compute. will compute this number of positive lags, and the same amount of negative lags,
                    for a total of 2*nlags+1 lags. At positive lags, var_one lags, at negative lags, var_one leads
        Returns:
        - corr_array: matrix of correlation coefficients [2*nlags+1 x lat x lon]
    """
    lags     = np.arange(-nlags,nlags+1,1)

    corr_mat = np.zeros((len(lags),var_one.shape[1],var_one.shape[2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lag in lags:
            var_one_temp = var_one.shift(time=lag)
            corr_mat[lag + nlags,:,:] = xr.corr(var_one_temp,var_two,dim = 'time')
    corr_array  = xr.DataArray(corr_mat,
                            coords={'lag': lags,'latitude': var_one.latitude,'longitude': var_two.longitude},
                            dims=['lag', 'latitude', 'longitude'])
    return(corr_array)
