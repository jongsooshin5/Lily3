import netCDF4 as nc
import numpy as np
import xarray as xr
import warnings

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

"""
A collection of tools to import and standardize data from FVCOM, radar, and ADCP. Assumed formats:
- FVCOM: Pickle (.p) file containing FVCOM data interpolated onto radar grid (regular grid)
- Radar: Pickle (.p) file containing regular grid of data at one level (surface)
- ADCP: MatLab (.mat) file containing time series of velocity components at deployment location
        (top sigma layer, corresponding w sigma = 0.85, is provided)

Provides functions:
    - read_file
    - norm_ds

TO-DO:

Lilli Enders (lilli.enders@outlook.com)
July 2023
"""

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

def norm_ds(dataset):
    # REMOVE CLIMATOLOGICAL MEAN
    # REMOVE SEASONAL CYCLE
    for i in range(12):
        m = dataset.sel(month=i)  # Get month
        m_norm = m.sla.data - np.nanmean(m.sla.data)  # Normalize w.r.t monthly av over ts
        m.sla.data = m_norm # Reassign to ds
    return dataset

def seasonal_detrend(dataset):
    """
    Remove seasonal trend from
    Inputs:
    - datatset: Dataset (xarray) containing cesm-le OR aviso data, in month/year format, with some SSH variable (here called sla)
                See pre-processing files to get data into this format
    Returns:
    - datatset: Dataset (xarray) or same type as input, with additional variable "sla_deseasonalized" containing output
    """
    ds = dataset
    sla_norm = np.zeros((ds.sla.shape))
    if 'Ensemble' in ds.sla.dims:
        monthly_avs = np.zeros((12,len(ds.latitude),len(ds.longitude)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for e in range(len(ds.Ensemble)):
                ds_temp = ds.sel(Ensemble=e)
                for mon in range(12):
                    m = ds_temp.sel(month=mon) # Get month I want
                    monthly_avs[mon] = np.nanmean(m.sla.data,axis=0)
                sla_norm[e,:] = ds_temp.sla.data - monthly_avs
            sla_norm = sla_norm - np.nanmean(sla_norm, axis=0) # Remove climatology
            ds["sla_deseasonalized"]=(['Ensemble','year', 'month', 'latitude', 'longitude'],  sla_norm)
    else:
        monthly_avs = np.zeros((12,len(ds.latitude),len(ds.longitude)))
        with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for mon in range(12):
                    m = ds.sel(month=mon) # Get month I want
                    monthly_avs[mon] = np.nanmean(m.sla.data,axis=0)
                sla_norm = ds.sla.data - monthly_avs
                dataset["sla_deseasonalized"]=(['year', 'month', 'latitude', 'longitude'],  sla_norm)
    dataset = ds
    return(dataset)

def fmt_time(dataset):
    ds = dataset
    year_start,year_end = ds.year.data[0],ds.year.data[-1]
    time = np.arange(np.datetime64(str(year_start) + '-01-01'), np.datetime64(str(year_end+1) + '-01-01'),
                     np.timedelta64(1, 'M'),  dtype='datetime64[M]')
    ds = ds.stack(time=['year', 'month'])
    ds = ds.drop_vars({'year', 'month', 'time'})
    ds = ds.assign_coords({'time': time})
    if 'Ensemble' in ds.sla.dims:
        ds = ds.transpose('Ensemble','time', 'latitude', 'longitude')
    else:
        ds = ds.transpose('time', 'latitude', 'longitude')
    dataset = ds
    return(dataset)

