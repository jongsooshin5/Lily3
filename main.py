from proj_utils import *
from tidy_data import *
import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings

"""
RUN ANALYSIS FOR OBS (ALTIMETRY)
"""
f_obs = '/Users/lillienders/Desktop/altimetry_sla_93_18.nc'
ds_obs = xr.open_dataset(f_obs) # Read netcdf data into xarray
ds_obs = seasonal_detrend(ds_obs)
gsi_lon_obs, gsi_lat_obs, sla_ts_obs, sla_ts_std_obs = gs_index(ds_obs)

plt.plot(sla_ts_obs)
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

