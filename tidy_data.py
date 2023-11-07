import numpy as np
import xarray as xr
import warnings

def read_file(file):
    """
    Read data from netCDF into xarray for analysis
    Inputs:
    - file: Path to netCDF file containing data, generated on vortex
    Outputs:
    - dataset: Dataset from file in xarray format
    """
    dataset = xr.open_dataset(file) # Read netcdf data into xarray
    return dataset

def fmt_time(dataset):
    """
    Reformat data set from (year, month, lat, lon) dimensions to (time, lat, lon) dimensions
    Inputs:
    - dataset: xarray dataset, read in from read_file with coordinates (year, month, lat, lon)
    Outputs:
    - dataset: original dataset, stacked into (time,lat,lon) format with time in datetime64 format
    """
    ds = dataset
    year_start,year_end = ds.year.data[0],ds.year.data[-1]
    time = np.arange(np.datetime64(str(year_start) + '-01-01'), np.datetime64(str(year_end+1) + '-01-01'),
                     np.timedelta64(1, 'M'),  dtype='datetime64[M]')
    ds = ds.stack(time=['year', 'month'])
    ds = ds.drop_vars({'year', 'month', 'time'})
    ds = ds.assign_coords({'time': time})
    ds = ds.transpose('time', 'latitude', 'longitude')
    dataset = ds
    return(dataset)

def rmv_clm(dataset):
    """
    Remove climatological mean (i.e., long-term average) from each datavariable in dataset
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
    - dataset: xarray dataset, formated in (time,lat,lon) dimensions with long-term mean removed
    """
    ds = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for var in list(ds.data_vars):
            ds[var] = ds[var] - np.nanmean(ds[var],axis=0)
    dataset = ds
    return(dataset)

def wind_pre_proc(dataset):
    ds = dataset
    # Reformat latitude and longitude to be consistent w observations
    ds = ds.sortby(ds.latitude)
    ds['longitude'] = ds['longitude'] + 360

    # Apply land sea mask
    ds['ewss'] = (ds['ewss'].where(ds.lsm < 0.1)) / (24 * 3600)
    ds['ewss'] = (ds['ewss'].where(ds.latitude < 80))
    ds['nsss'] = (ds['nsss'].where(ds.lsm < 0.1)) / (24 * 3600)
    ds['nsss'] = (ds['nsss'].where(ds.latitude < 80))

    dataset = ds
    return (dataset)

def seasonal_detrend(dataset):
    """
    Remove seasonal cycle (here defined as monthly mean) from time series of each variable
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
    - dataset: xarray dataset, formated in (time,lat,lon) dimensions with seasonal cycle removed
    """
    ds = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for var in list(ds.data_vars):
            mn_av = np.zeros((12,len(ds.latitude),len(ds.longitude)))
            mn_av[:] = np.nan
            for mn in range(12):
                mn_av[mn,:,:] = np.nanmean(ds[var][mn::12,:,:],axis=0)
            mn_av = np.tile(mn_av, (int(len(ds[var])/12),1,1))
            detrend_temp = ds[var] - mn_av
            ds[var] = detrend_temp
    dataset = ds
    return(dataset)

def linear_detrend(dataset):
    """
    Remove a linear trend from each grid point
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
     - dataset: xarray dataset, formated in (time,lat,lon) dimensions with linear trend removed
    """
    ds = dataset
    ds_poly = ds.polyfit(dim='time', deg = 1)
    indices = np.arange(len(ds.time))
    for var in list(ds.data_vars):
        fit_string = var+'_polyfit_coefficients'
        slope = np.array(ds_poly[fit_string][0]).flatten()
        intercept = np.array(ds_poly[fit_string][1]).flatten()
        lin_fit = np.zeros((len(ds.time),len(slope)))
        for loc in range(len(slope)):
            lin_fit[:,loc] = slope[loc]*indices + intercept[loc]
        lin_fit = np.reshape(lin_fit, (len(ds.time),len(ds.latitude),len(ds.longitude)))
        detrended_series = ds[var]-lin_fit
        ds[var] = detrended_series
    dataset = ds
    return(ds)

def tidy_read(file):
    """
    Performs whole cleaning/detrending routine given file
    Inputs:
    - file: Path to netCDF file containing data, generated on vortex
    Outputs:
    - ds: xarray dataset, formated in (time,lat,lon) dimensions with climatology, seasonal trend, and linear trend removed
    """
    ds = read_file(file)
    ds = fmt_time(ds)
    ds = rmv_clm(ds)
    ds = seasonal_detrend(ds)
    ds = linear_detrend(ds)
    return ds
