from fmcal import fm_cal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import time

cal = fm_cal('../test/test_constcd_config.ini')
cal.initialize()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()
for cdind, cd in enumerate(np.arange(cal.cdmin, cal.cdmax, cal.cdinc)):
    simres = cal.update_const(cdind, cd)

    ax.plot(simres[:cdind,0], simres[:cdind,1], '-o')  # fit the line
    plt.pause(3)
    plt.draw()
    # fig.canvas.draw()  # draw
    # time.sleep(0.5)  # sleep

tmp = 0