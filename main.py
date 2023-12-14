from run_cesm_lens import *
from run_altimetry import *
from run_cesm_ihesp import *

gsi_sd_alt,  damp_t_alt, crossing_alt,vars_alt = run_altimetry()

gsi_sd_lens, damp_t_lens,crossing_lens,vars_lens = run_cesm_lens()
gsi_sd_ihesp,  damp_t_ihesp,crossing_ihesp,vars_ihesp  = run_cesm_ihesp()

print(crossing_lens)
print(vars_lens)
#print(crossing_alt)
#print(vars_alt)
#print(crossing_ihesp)
#print(vars_ihesp)
print('Done!')


fig = plt.figure(figsize=(20,8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))

ax1.hist([crossing_ihesp[:,0], crossing_lens[:,0]],color=['darkred','darksalmon'],label=['CESM-HR','CESM-LE'],alpha=0.6,stacked=True)
ax1.hist(crossing_alt[0],color='k',rwidth = 5,label='Altimetry',bins=1,alpha=1)


ax2.hist([vars_ihesp[:,0]*100, vars_lens[:,0]*100],color=['darkred','darksalmon'],label=['CESM-HR','CESM-LE'],alpha=0.6,stacked=True)
ax2.hist(vars_alt[0]*100,color='k',rwidth = 5,label='Altimetry',bins=1,alpha=1)


ax1.set_title('Number of Zero Crossings: Fist EOF of GSI',fontsize=18)
ax1.set_xlabel('E-Number of Zero Crossings',fontsize=18)
ax1.set_ylabel('Bin Count', fontsize=18)
ax1.legend(fontsize=15,loc='upper center')
ax1.yaxis.set_tick_params(labelsize=15)
ax1.xaxis.set_tick_params(labelsize=15)

ax2.set_title('Percent Variance Explained: First EOF of GSI',fontsize=18)
ax2.set_xlabel('Percent Variance',fontsize=18)
ax2.set_ylabel('Bin Count', fontsize=18)
ax2.legend(fontsize=15,loc='upper center')
ax2.yaxis.set_tick_params(labelsize=15)
ax2.xaxis.set_tick_params(labelsize=15)
plt.show()


fig = plt.figure(figsize=(20,8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))

ax1.hist([crossing_ihesp[:,1], crossing_lens[:,1]],color=['darkred','darksalmon'],label=['CESM-HR','CESM-LE'],alpha=0.6,stacked=True)
ax1.hist(crossing_alt[1],color='k',rwidth = 5,label='Altimetry',bins=1,alpha=1)


ax2.hist([vars_ihesp[:,1]*100, vars_lens[:,1]*100],color=['darkred','darksalmon'],label=['CESM-HR','CESM-LE'],alpha=0.6,stacked=True)
ax2.hist(vars_alt[1]*100,color='k',rwidth = 5,label='Altimetry',bins=1,alpha=1)


ax1.set_title('Number of Zero Crossings: Second EOF of GSI',fontsize=18)
ax1.set_xlabel('E-Number of Zero Crossings',fontsize=18)
ax1.set_ylabel('Bin Count', fontsize=18)
ax1.legend(fontsize=15,loc='upper center')
ax1.yaxis.set_tick_params(labelsize=15)
ax1.xaxis.set_tick_params(labelsize=15)

ax2.set_title('Percent Variance Explained: Second EOF of GSI',fontsize=18)
ax2.set_xlabel('Percent Variance',fontsize=18)
ax2.set_ylabel('Bin Count', fontsize=18)
ax2.legend(fontsize=15,loc='upper center')
ax2.yaxis.set_tick_params(labelsize=15)
ax2.xaxis.set_tick_params(labelsize=15)
plt.show()


fig = plt.figure(figsize=(20,8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))

ax1.hist([crossing_ihesp[:,2], crossing_lens[:,0]],color=['darkred','darksalmon'],label=['CESM-HR','CESM-LE'],alpha=0.6,stacked=True)
ax1.hist(crossing_alt[2],color='k',rwidth = 5,label='Altimetry',bins=1,alpha=1)


ax2.hist([vars_ihesp[:,2]*100, vars_lens[:,2]*100],color=['darkred','darksalmon'],label=['CESM-HR','CESM-LE'],alpha=0.6,stacked=True)
ax2.hist(vars_alt[2]*100,color='k',rwidth = 5,label='Altimetry',bins=1,alpha=1)


ax1.set_title('Number of Zero Crossings: Third EOF of GSI',fontsize=18)
ax1.set_xlabel('E-Number of Zero Crossings',fontsize=18)
ax1.set_ylabel('Bin Count', fontsize=18)
ax1.legend(fontsize=15,loc='upper center')
ax1.yaxis.set_tick_params(labelsize=15)
ax1.xaxis.set_tick_params(labelsize=15)

ax2.set_title('Percent Variance Explained: Third EOF of GSI',fontsize=18)
ax2.set_xlabel('Percent Variance',fontsize=18)
ax2.set_ylabel('Bin Count', fontsize=18)
ax2.legend(fontsize=15,loc='upper center')
ax2.yaxis.set_tick_params(labelsize=15)
ax2.xaxis.set_tick_params(labelsize=15)
plt.show()