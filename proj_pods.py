import numpy as np
import warnings
def var_magnitude(dataset, gsi_lon, gsi_lat):
    ds = dataset
    std_gsi = np.zeros((len(gsi_lon)))
    std_cm  = np.nanstd(ds.sla,axis=0)*100
    for x in range(len(gsi_lon)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            std_gsi[x] = std_cm[np.nanargmin(abs(ds.latitude.data - gsi_lat[x])), np.nanargmin(abs(ds.longitude.data - gsi_lon[x]))]
    mn_std_gsi = np.nanmean(std_gsi)
    return(mn_std_gsi)

def damping_time_scale(acf):
    efold = 1/np.exp(1)
    find_crossing = acf - efold
    damping_t = np.where(np.diff(np.sign(find_crossing)))[0][0]+1
    return(damping_t)