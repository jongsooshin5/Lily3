import matplotlib.pyplot as plt
import cartopy as cartopy
#import cartopy.feature as cfeature
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


def init_map(bbox, crs=ccrs.PlateCarree(), ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    # fig = plt.gcf()

    # ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    # ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(bbox, crs)

    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.LAND,facecolor='k',zorder=-1)

    # Add Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle=':')
    gl.top_labels = gl.right_labels = False

    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')

    return ax
