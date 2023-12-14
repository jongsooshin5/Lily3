import numpy as np
import warnings
def var_magnitude(dataset, gsi_lon, gsi_lat):
    ds = dataset
    std_gsi = np.zeros((len(gsi_lon)))
    std_cm  = np.nanstd(ds.sla,axis=0)
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

def eof_crossings(eofs_gsi, per_var_gsi):
    mode_one = np.array([((eofs_gsi.data[0][:-1] * eofs_gsi.data[0][1:]) < 0).sum(),per_var_gsi[0]])
    mode_two = np.array([((eofs_gsi.data[1][:-1] * eofs_gsi.data[1][1:]) < 0).sum(),per_var_gsi[1]])
    mode_three = np.array([((eofs_gsi.data[2][:-1] * eofs_gsi.data[2][1:]) < 0).sum(),per_var_gsi[2]])
    crossings = np.array([mode_one[0],mode_two[0],mode_three[0]])
    vars      = np.array([mode_one[1],mode_two[1],mode_three[1]])
    return(crossings, vars)