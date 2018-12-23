from fmcal import fm_cal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import plotly.plotly as py
import plotly.graph_objs as go
import time

cal = fm_cal('../test/test_constcd_config.ini')
cal.initialize()
#cal.create_ini_file()

fig = plt.figure()
ax = fig.add_subplot(111)
# cax = make_axes_locatable(ax).append_axes('right', size='5%', pad='2%')
plt.ion()

fig.show()
fig.canvas.draw()
x = np.arange(cal.mcdmin['0'], cal.mcdmax['0'], cal.mcdinc['0'])
numx = len(x)

tcd ={}
tcdind = {}
tcount = 0

for cdind0, cd0 in enumerate(x):
    tcd[0] = cd0
    tcdind[0] = cdind0
    simres = cal.update_var(tcount, tcd)
    # if tcount > 0:
    ax.plot(simres.loc[:cdind0, 'cd0'], simres.loc[:cdind0, 'rmse'], '-o')  # fit the line

    ax.set_xlabel('cd0')
    ax.set_ylabel('RMSE')

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