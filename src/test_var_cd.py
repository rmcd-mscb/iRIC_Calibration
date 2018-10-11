from fmcal import fm_cal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import plotly.plotly as py
import plotly.graph_objs as go
import time

cal = fm_cal('../test/test_varcd_config.ini')
cal.initialize()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = make_axes_locatable(ax).append_axes('right', size='5%', pad='2%')
plt.ion()

fig.show()
fig.canvas.draw()
x = np.arange(cal.mcdmin['0'], cal.mcdmax['0'], cal.mcdinc['0'])
numx = len(x)
y = np.arange(cal.mcdmin['1'], cal.mcdmax['1'], cal.mcdinc['1'])
numy = len(y)
X, Y = np.meshgrid(x, y, indexing='ij')
tcd ={}
tcdind = {}
tcount = 0
tmpz = np.zeros(shape=(numx,numy))
for cdind0, cd0 in enumerate(x):
    tcd[0] = cd0
    tcdind[0] = cdind0
    for cdind1, cd1 in enumerate(y):
        tcd[1] = cd1
        tcdind[1] = cdind1
        numind = len(tcdind)
        # simres = cal.update_var(cdind0, cd0, cdind1, cd1)
        simres = cal.update_var2(tcount, tcd)
        tmpz[cdind0,cdind1] = simres.loc[tcount, 'rmse']
        # tmpz =np.reshape(simres['rmse'].values,(numx,numy))
        cs = ax.contourf(X, Y, tmpz, 20, cmap=plt.cm.bone_r)
        ax.set_xlabel('cd0')
        ax.set_ylabel('cd1')
        cbar = fig.colorbar(cs,cax=cax)
        cbar.ax.set_ylabel('RMSE')
        #     ax.plot(simres[:cdind,0], simres[:cdind,1], '-o')  # fit the line
        fig.canvas.update()
        fig.canvas.flush_events()
        plt.pause(1)
        plt.draw()
        tcount+=1
#     fig.canvas.draw()   # draw
#     time.sleep(0.5)    #sleep
plt.savefig('testfig.png', dpi=300)
cal.resdf.to_csv(cal.rmse_file)
tmp = 0