import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import numpy as np
import cmocean as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

sns.set()


def ts_plot(x_data, y_data, label1='', x_data_2=[], y_data_2=[], label2='', xlab='', ylab='', title=None, save=False,
            sv_pth=None, sv_name=None):
    fig = plt.figure(figsize=(10, 6))

    zline = plt.axhline(0, color='k', linewidth=2)
    feat = plt.plot(x_data, y_data, color='k', linewidth=1.5, label=label1)
    feat2 = plt.plot(x_data_2, y_data_2, color='r', linewidth=1.5, label=label2)

    leg = plt.legend(fontsize=18)
    ylab = plt.ylabel(ylab, fontsize=18)
    xtix = plt.yticks(fontsize=18)
    ytix = plt.xticks(fontsize=18)

    ylim = plt.ylim([-np.max(abs(y_data)) - 0.1, np.max(abs(y_data)) + 0.1])
    xlim = plt.xlim([np.min(x_data), np.max(x_data)])

    tit = plt.title(title, fontsize=20)
    if save == True:
        plt.savefig(sv_pth + sv_name + '.png', format='png', bbox_inches="tight",dpi=500)
    plt.close()

def acf_plot(acf, n_eff,save=False,sv_pth=None, sv_name=None):
    fig = plt.figure(figsize=(12, 6))

    z_line = plt.axhline(0, color='r', linewidth=2)
    markerline, stemlines, baseline = plt.stem(np.arange(0, 60), acf, linefmt='k')
    face = markerline.set_markerfacecolor('k')
    edge = markerline.set_markeredgecolor('k')
    n_bot = plt.plot(np.arange(-1, 61), -1.96 / np.sqrt(n_eff[0:62]), color='k', ls='--')
    n_top = plt.plot(np.arange(-1, 61), 1.96 / np.sqrt(n_eff[0:62]), color='k', ls='--')

    ylim = plt.ylim(-1.1, 1.1)
    xlim = plt.xlim(-1, 60)

    xlab = plt.xlabel('Lag (Months) ', fontsize=18)
    ylab = plt.ylabel('ACF', fontsize=18)
    ytix = plt.yticks(fontsize=18)
    xtix = plt.xticks(fontsize=18)
    if save == True:
        plt.savefig(sv_pth + sv_name + '.png', format='png', bbox_inches="tight",dpi=500)
    plt.close()

def spatial_plot(x_data, y_data, z_data, bthy_data=None, levels=None, x_gsi=None, y_gsi=None, region='NA',
                 add_bathy=False, add_gsi=False,save=False,sv_pth=None, sv_name=None):
    # Choose region to plot
    if region == 'GS':
        bbox = [280, 308, 33, 46]
    if region == 'NA':
        bbox = [260, 360, 0, 90]

    fig = plt.figure(figsize=(14, 8))
    crs = ccrs.PlateCarree()
    ax = plt.subplot(1, 1, 1, projection=crs)
    ax.set_extent(bbox, crs)

    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='k', zorder=1)

    # Add Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')

    # levels = np.linspace(0,50,25)
    colorplot = plt.contourf(x_data, y_data, z_data, levels=50, cmap='Spectral_r', zorder=0)
    max_data = int(round(np.nanmax(abs(z_data)), -1))
    colorplot.set_clim(0, max_data)

    if add_bathy == True:
        if levels == None:
            max_bathy = int(round(np.nanmax(abs(np.nanmean(bthy_data, axis=0))), -1))
            levels = np.linspace(0, max_bathy, int(max_bathy / 10) + 1)
        contours = plt.contour(x_data, y_data, abs(bthy_data), levels=levels, colors='k', zorder=10, linewidths=0.75)
        labels = plt.clabel(contours, inline=True, fontsize=10)

    if add_gsi == True:
        plt.scatter(x_gsi, y_gsi, color='k')

    plt.xlabel('Latitude ', fontsize=25)
    plt.ylabel('Longitude', fontsize=25)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    # plt.title('Standard Deviation: Altimetry (1992-2012)',fontsize=20)

    cbar = plt.colorbar(colorplot, fraction=0.02, pad=0.04, ticks=np.linspace(0, max_data, int(max_data / 10) + 1))
    cbar.set_label('Standard Deviation [cm]', size='20', labelpad=25)
    cbar.ax.tick_params(labelsize=20)
    if save == True:
        plt.savefig(sv_pth + sv_name + '.png', format='png', bbox_inches="tight",dpi=500)
    plt.close()

def spatial_plot_div(x_data, y_data, z_data, label='EOF', bthy_data=None, levels=None, region='GS', add_bathy=False,
                     add_gsi=False,save=False,sv_pth=None, sv_name=None):
    # Choose region to plot
    if region == 'GS':
        bbox = [280, 308, 33, 46]
    if region == 'NA':
        bbox = [260, 360, 0, 90]

    fig = plt.figure(figsize=(14, 8))
    crs = ccrs.PlateCarree()
    ax = plt.subplot(1, 1, 1, projection=crs)
    ax.set_extent(bbox, crs)

    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='k', zorder=1)

    # Add Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')

    # max_data  =  round(np.nanmax(abs(z_data)),3)
    max_data = 0.11
    levels = np.linspace(-max_data, max_data, 39)
    colorplot = plt.contourf(x_data, y_data, z_data, levels=levels, cmap='cmo.balance', zorder=0)
    colorplot.set_clim(-max_data, max_data)

    if add_bathy == True:
        contours = plt.contour(x_data, y_data, abs(bthy_data), levels=levels, colors='k', zorder=10, linewidths=0.75)
        labels = plt.clabel(contours, inline=True, fontsize=10)

    plt.xlabel('Latitude ', fontsize=25)
    plt.ylabel('Longitude', fontsize=25)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    # plt.title('Standard Deviation: Altimetry (1992-2012)',fontsize=20)

    cbar = plt.colorbar(colorplot, fraction=0.02, pad=0.04, ticks=np.linspace(-max_data, max_data, 9))
    ticks = cbar.get_ticks()
    ticklabels = [f'{tick:.{2}f}' for tick in ticks]
    cbar.set_ticklabels(ticklabels)

    cbar.set_label(label, size='20', labelpad=25)
    cbar.ax.tick_params(labelsize=20)
    if save == True:
        plt.savefig(sv_pth + sv_name + '.png', format='png', bbox_inches="tight",dpi=500)
    plt.close()

def spatial_scatter(x_data, y_data, z_data, label='EOF', region='GS',save=False,sv_pth=None, sv_name=None):
    # Choose region to plot
    if region == 'GS':
        bbox = [280, 308, 33, 46]
    if region == 'NA':
        bbox = [260, 360, 0, 90]

    fig = plt.figure(figsize=(14, 8))
    crs = ccrs.PlateCarree()
    ax = plt.subplot(1, 1, 1, projection=crs)
    ax.set_extent(bbox, crs)

    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='k', zorder=1)

    # Add Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')

    plt.xlabel('Latitude ', fontsize=25)
    plt.ylabel('Longitude', fontsize=25)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}

    max_data = 0.40
    scat = plt.scatter(x_data, y_data, s=abs(z_data * 500), c=z_data, vmin=-max_data, vmax=max_data, cmap='cmo.balance')

    # plt.title('Standard Deviation: Altimetry (1992-2012)',fontsize=20)
    ax.imshow(np.tile(np.array([[[255, 255, 255]]],
                               dtype=np.uint8), [2, 2, 1]),
              origin='upper',
              transform=ccrs.PlateCarree(),
              extent=[-180, 180, -90, 90])

    cbar = plt.colorbar(scat, fraction=0.02, pad=0.04, ticks=np.linspace(-max_data, max_data, 9))
    ticks = cbar.get_ticks()
    ticklabels = [f'{tick:.{2}f}' for tick in ticks]
    cbar.set_ticklabels(ticklabels)
    cbar.set_label(label, size='20', labelpad=25)
    cbar.ax.tick_params(labelsize=20)
    if save == True:
        plt.savefig(sv_pth + sv_name + '.png', format='png', bbox_inches="tight",dpi=500)
    plt.close()
def eof_longitude(gsi_lon,eofs_gsi,save=False,sv_pth=None, sv_name=None):
    fig    = plt.figure(figsize=(12,6))
    eof1   = plt.plot(gsi_lon,eofs_gsi[0],color = 'k', label='EOF 1')
    eof2   = plt.plot(gsi_lon,eofs_gsi[1],color = 'r', label='EOF 2')
    eof3   = plt.plot(gsi_lon,eofs_gsi[2],color = 'g', label='EOF 3')
    eof4   = plt.plot(gsi_lon,eofs_gsi[3],color = 'b', label='EOF 4')

    ylab  = plt.ylabel('EOF Magnitude', fontsize=18)
    xlab  = plt.xlabel('Longitude (˚)', fontsize=18)

    xtix  = plt.yticks(fontsize=18)
    ytix  = plt.xticks(fontsize=18)

    ylim  = plt.ylim([-np.max(abs(eofs_gsi[0]))-0.1, np.max(abs(eofs_gsi[0]))+0.1])
    xlim  = plt.xlim([np.min(gsi_lon[0]), np.max(gsi_lon[-1])])

    zline  = plt.axhline(0,color='k', linewidth=2)
    leg    = plt.legend()
    if save == True:
        plt.savefig(sv_pth + sv_name + '.png', format='png', bbox_inches="tight",dpi=500)
    plt.close()