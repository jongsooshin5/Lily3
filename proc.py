import netCDF4 as nc
import numpy as np
import xarray as xr
import warnings
from eofs.xarray import Eof
import metpy.calc as mpcalc

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
def gs_index(dataset):
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
        if 'Ensemble' in ds.sla.dims:
            # Take time average to produce a single field
            mn_std = ds.sla_std.data
            mn_sla = np.nanmean(ds.sla.data, axis=(1, 2))
            # Calculate location (latitude) of maximum standard deviation
            gsi_lat = np.array(np.zeros((len(ds.Ensemble), len(ds.longitude))), dtype=int)
            gsi_lat_idx = np.array(np.zeros((len(ds.Ensemble), len(ds.longitude))), dtype=int)
            for ens in range(len(ds.Ensemble)):
                gsi_lat_idx[ens, :] = np.nanargmax(mn_std[ens, :, :], axis=0)
                gsi_lat[ens, :] = ds.latitude[gsi_lat_idx[ens, :]].data
            sla_flt = ds.sla_deseasonalized.data.reshape(len(ds.Ensemble), len(ds.year) * len(ds.month), len(ds.latitude),
                                          len(ds.longitude))
            sla_ts = np.zeros((len(ds.Ensemble), len(ds.year) * len(ds.month)))
            sla_ts_std = np.zeros((len(ds.Ensemble), len(ds.year) * len(ds.month)))
            for ens in range(len(ds.Ensemble)):
                temp = np.zeros(len(ds.longitude))
                for t in range((len(ds.year) * len(ds.month))):
                    for lon in range(len(ds.longitude)):
                        temp[lon] = sla_flt[ens, t, gsi_lat_idx[ens, lon], lon]
                    sla_ts[ens, t] = np.nanmean(temp)
                    sla_ts_std[ens, t] = np.nanstd(temp)
        else:
            # Take time average to produce a single field
            mn_std = np.nanmean(ds.sla_std.data, axis=(0, 1))
            mn_sla = np.nanmean(ds.sla.data, axis=(0, 1))
            # Calculate location (latitude) of maximum standard deviation
            gsi_lat_idx = np.nanargmax(mn_std, axis=0)
            gsi_lat = ds.latitude[gsi_lat_idx].data
            sla_flt = ds.sla_deseasonalized.data.reshape(len(ds.year) * len(ds.month), len(ds.latitude), len(ds.longitude))
            temp = np.zeros(len(ds.longitude))
            sla_ts = np.zeros(len(ds.year) * len(ds.month))
            sla_ts_std = np.zeros(len(ds.year) * len(ds.month))

            for t in range((len(ds.year) * len(ds.month))):
                for lon in range(len(ds.longitude)):
                    temp[lon] = sla_flt[t, gsi_lat_idx[lon], lon]
                sla_ts[t] = np.nanmean(temp)
                sla_ts_std[t] = np.nanstd(temp)
    return (gsi_lon, gsi_lat, sla_ts, sla_ts_std)

def calc_eofs(array,num_modes = 1):
    coslat = np.cos(np.deg2rad(array.coords['latitude'].values))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(array, weights=wgts)
    eofs = np.squeeze(solver.eofs(neofs=num_modes))
    pcs = np.squeeze(solver.pcs(npcs=num_modes, pcscaling=1))
    per_var = solver.varianceFraction()
    return(eofs, pcs, per_var)
