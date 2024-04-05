from proj_utils import *
from tidy_data import *
from proj_pods import *
from viz import *
import numpy as np
import datetime as dt
import xarray as xr
import warnings
def run_altimetry():
    fig_pth = '/Users/lillienders/Desktop/First Generals/Figures/Altimetry/'
    fig_save = True
    """
    RUN ANALYSIS FOR OBS (ALTIMETRY)
    """
    # region: Calculations and Plots

    f = '/Users/lillienders/Desktop/First Generals/Data/Observations/altimetry_sla_93_22.nc'
    ds = tidy_read(f)
    ds['sla'] = ds['sla'] * 100
    ds['adt'] = (['latitude', 'longitude'], get_contours(f))
    ds['adt'] = ds['adt']*100
    ds['sla_std'] = (['latitude', 'longitude'], np.nanmean(ds.sla_std.data, axis=0))
    ds['sla_std'] = ds['sla_std']*100
    #ds = ds.sel(longitude=slice(279, 310), latitude=slice(32, 46), time=slice('1993-01-01', '2022-12-01'))
    ds.to_netcdf('altimetric_ssh_clean.nc')
    # Calculate GSI using Joyce method
    gsi_lon_joyce, gsi_lat_joyce, sla_ts_joyce, sla_ts_std_joyce = gs_index_joyce(ds)
    gsi_norm_joyce = (sla_ts_joyce - np.nanmean(sla_ts_joyce))/sla_ts_std_joyce

    # Calculate GSI using isoline method
    gsi_lon, gsi_lat, sla_ts, sla_ts_std = gs_index(ds,ds['adt'], alt = True)
    gsi_norm = (sla_ts - np.nanmean(sla_ts))/np.nanstd(sla_ts)

    plt.plot(gsi_norm)
    plt.show
    gsi_array = xr.DataArray(
        data=gsi_norm,
        dims=['time'],
        name = 'gsi',
        coords=dict(
            time      = ds.time.data
        ),
        attrs=dict(
            description="GSI, 1993-2022",
            units="-",
        ),
    )
    gsi_array.to_netcdf('/Users/lillienders/Desktop/First Generals/Data/Observations/gsi_array_2.nc')

    gsi_annual = smooth_data(gsi_norm)
    gsi_years  = []
    for yr in range(1,26):
        gsi_years.append(dt.datetime(int(1993+yr),1,1))

    # Correlate GSI methods
    gsi_corr = np.corrcoef(gsi_norm,gsi_norm_joyce)[0,1]

    # Calculate ACF, number of effective degrees of freedom
    acf, n_eff = get_acf(gsi_norm)


    # Calculate EOFs: Full spatial domain
    num_modes = 4
    bbox      = [280, 308, 33, 46]
    adt_slice = ds['adt'][np.nanargmin(abs(ds.latitude-bbox[2])):np.nanargmin(abs(ds.latitude-bbox[3])),
                         np.nanargmin(abs(ds.longitude-bbox[0])):np.nanargmin(abs(ds.longitude-bbox[1]))]
    gs_slice  = ds.isel(longitude = slice(np.nanargmin(abs(ds.longitude-bbox[0])), np.nanargmin(abs(ds.longitude-bbox[1]))),
                        latitude  = slice(np.nanargmin(abs(ds.latitude-bbox[2])), np.nanargmin(abs(ds.latitude-bbox[3]))))
    eofs, pcs, per_var = calc_eofs(gs_slice['sla'],num_modes)
    pcs  = - pcs
    eofs = - eofs

    # Calculate EOFS: At GSI array
    sla_gsi = np.zeros((len(ds.time),len(gsi_lon)))
    for t in range(len(ds.time)):
        for x in range(len(gsi_lon)):
            sla_gsi[t,x] = ds['sla'][t,np.nanargmin(abs(ds.latitude.data - gsi_lat[x])), np.nanargmin(abs(ds.longitude.data - gsi_lon[x]))]

    sla_gsi_array = xr.DataArray(sla_gsi,
                                 coords = {'time': ds.time,'lon': gsi_lat},
                                 dims   = ['time', 'lon'])

    eofs_gsi, pcs_gsi, per_var_gsi = calc_eofs(sla_gsi_array,num_modes)
    acf_spatial, n_eff_spatial = get_acf(np.nanmean(sla_gsi,axis=0))
    #acf_spatial, n_eff_spatial = get_acf(eofs_gsi[0,:])

    # end region
    # region: Plots
    # Figure (1): Spatial plot of standard deviation with maximum standard deviation isoline and location of GSI array
    spatial_plot(ds.longitude, ds.latitude, np.nanstd(ds.sla,axis=0),bthy_data= abs(ds['adt']),
                 levels = [get_max_contour(ds,ds['adt'],contours_to_try = np.linspace(30, 50, 21))], x_gsi = gsi_lon, y_gsi = gsi_lat,
                 region='GS', add_gsi = True,add_bathy=True, save=fig_save,sv_pth=fig_pth, sv_name='alt_spatial_std_gsi')

    # Figure (2): Time series of monthly and yearly GSI (isoline method)
    ts_plot(ds.time, gsi_norm, label1 = 'Monthly Index', x_data_2 = gsi_years, y_data_2 = gsi_annual,label2 = 'Annual Index',
            xlab = 'Year', ylab = 'GSI', title = 'Normalized GSI',save=fig_save,sv_pth=fig_pth, sv_name='alt_gsi_monthly_annual')

    # Figure (3): Time series of GSI from Joyce and isoline methods
    ts_plot(ds.time, gsi_norm, label1 = 'Isoline Method', x_data_2 = ds.time, y_data_2 = gsi_norm_joyce,label2 = 'Joyce Method',
            xlab = 'Year', ylab = 'GSI', title = 'GSI Method Comparison (r = ' + '%1.3f' % gsi_corr + ')',
            save=fig_save,sv_pth=fig_pth, sv_name='alt_gsi_joyce_isoline')

    # Figure (4): ACF of GSI timeseries
    acf_plot(acf,n_eff,save=fig_save,sv_pth=fig_pth, sv_name='alt_acf')

    acf_plot(acf_spatial,n_eff_spatial,save=fig_save,sv_pth=fig_pth, sv_name='alt_acf_spatial')

    # Figure (5): Spatial maps of EOFs
    # Figure (6): Time series of principal components, GSI
    # Figure (7): Spatial maps of EOFs at GSI array
    # Figure (8): Time series of principal components at GSI array, GSI

    for eof in range(num_modes):
        pc_to_plot = eof+1
        corr = np.corrcoef(pcs[:,pc_to_plot-1],gsi_norm)[0,1]
        corr_gsi = np.corrcoef(pcs_gsi[:, pc_to_plot - 1], gsi_norm)[0, 1]

        spatial_plot_div(gs_slice.longitude, gs_slice.latitude, eofs[pc_to_plot-1], label = 'EOF' + str(pc_to_plot), bthy_data= abs(adt_slice)*100,
                    title='EOF' + str(pc_to_plot) + '(% Var = ' + '%1.2f' % (per_var[pc_to_plot - 1].data * 100) + ')',
                     levels = 10, add_bathy=fig_save, save=True,sv_pth=fig_pth, sv_name='alt_spatial_eof_' + str(pc_to_plot))

        ts_plot(ds.time, pcs[:,pc_to_plot-1],label1 = 'PC' + str(pc_to_plot),x_data_2 = ds.time, y_data_2 = gsi_norm, label2= 'GSI',
                xlab = 'Year', ylab = 'PC' + str(pc_to_plot), title = 'PC' +str(pc_to_plot) + ' v GSI (r = ' + '%1.3f' % corr + ')',
                save=fig_save,sv_pth=fig_pth, sv_name='alt_tseries_pc_' + str(pc_to_plot))

        spatial_scatter(gsi_lon,gsi_lat,eofs_gsi[pc_to_plot-1],label = 'EOF' + str(pc_to_plot),
                        title='EOF' + str(pc_to_plot) + '(% Var = ' + '%1.2f' % (per_var_gsi[pc_to_plot - 1].data * 100) + ')',
                        save=fig_save,sv_pth=fig_pth, sv_name='alt_spatial_gsi_eof_' + str(pc_to_plot))


        ts_plot(ds.time, pcs_gsi[:,pc_to_plot-1],label1 = 'PC' + str(pc_to_plot),x_data_2 = ds.time, y_data_2 = gsi_norm, label2= 'GSI',
                xlab = 'Year', ylab = 'PC' + str(pc_to_plot), title = 'PC' +str(pc_to_plot) + ' v GSI (r = ' + '%1.3f' % corr_gsi + ')',
                save=fig_save,sv_pth=fig_pth, sv_name='alt_tseries_gsi_pc_' + str(pc_to_plot))

    eof_longitude(gsi_lon,eofs_gsi, save=fig_save,sv_pth=fig_pth, sv_name='alt_eof_longitude_plt')
    # end region
    # region: PODS
    alt_gsi_sd = var_magnitude(ds,gsi_lon,gsi_lat)
    alt_damp_t  = damping_time_scale(acf)
    alt_damp_s = damping_spatial_scale(acf_spatial)
    alt_cross, alt_var   = eof_crossings(eofs_gsi, per_var_gsi)
    # end region
    return(alt_gsi_sd,alt_damp_t,alt_damp_s,alt_cross,alt_var,acf_spatial,n_eff_spatial)
