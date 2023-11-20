import netCDF4 as nc
import numpy as np
import xarray as xr
import warnings
from eofs.xarray import Eof
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import statsmodels.api as sm

def calc_wsc(dataset, u_stress, v_stress):
    """
    Computes wind stress curl from u/v components of wind stress
    Inputs:
    - dataset: xarray dataset containing time, latitude, and longitude dimensions that correspond to wind stress data
    - u_stress: xarray data array containing eastward (x-direction) turbulent surface stress (wind stress)
    - v_stress: xarray data array containing northward (y-direction) turbulent surface stress (wind stress)
    Outputs:
    - ds_wsc: xarray dataset containing data variable 'wsc' which is the curl of the winds stress vector provided
    """
    ds = dataset
    tauy = v_stress.metpy.assign_crs(
        grid_mapping_name='latitude_longitude',
        earth_radius=6371229.0
    )
    taux = u_stress.metpy.assign_crs(
        grid_mapping_name='latitude_longitude',
        earth_radius=6371229.0
    )
    taux.attrs['units'] = "m/s"
    tauy.attrs['units'] = "m/s"
    wind_curl = mpcalc.vorticity(taux,tauy)
    wind_curl = np.array(wind_curl)
    wind_curl[wind_curl > 3e-7] = 3e-7
    wind_curl[wind_curl < -3e-7] = -3e-7
    wind_curl[wind_curl < -3e-7] = -3e-7
    ds_wsc = xr.Dataset(
        data_vars=dict(
            wsc = (['time','latitude','longitude'], wind_curl)),
        coords=dict(
            time      = ds.time.data,
            latitude  = ds.latitude.data,
            longitude = ds.longitude.data,
        ))
    return(ds_wsc)

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
        mn_std = np.nanmean(ds.sla_std.data, axis=0)
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


def gs_index(dataset, adt):
    """
    Get Gulf Stream Index Locations (Contour w Max Standard Deviation)
    Inputs:
    - dataset: containing longitude, latitude, sla, sla_std
    - adt: dynamic topography contours
    Returns:
    - gsi_lon: longitudes of gulf stream index points
    - gsi_lat: latitudes of gulf stream index points
    - std_ts: time series of gulf stream index
    - sla_ts_std: standard deviation of std_ts
    """
    ds = dataset
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


def get_max_contour(dataset, adt):
    ds = dataset
    std_field = np.nanstd(ds.sla, axis=0) * 100
    contours_to_try = np.linspace(27, 49, 23)
    std_contours = np.zeros(len(contours_to_try))
    for c in range(len(contours_to_try)):
        x_temp, y_temp = get_contour_info(ds.longitude, ds.latitude, abs(adt), contour_to_get=int(contours_to_try[c]))
        temp = 0
        for x in range(len(x_temp)):
            temp = temp + std_field[
                np.nanargmin(abs(ds.latitude.data - y_temp[x])), np.nanargmin(abs(ds.longitude.data - x_temp[x]))]
        std_contours[c] = temp
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

    for item in contours.collections:
        for i in item.get_paths():
            v = i.vertices
            x = v[:, 0]
            y = v[:, 1]
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