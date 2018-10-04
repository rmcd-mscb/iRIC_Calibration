from FMCal import fm_cal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import time

cal = fm_cal('./test/test_constcd_config.ini')
cal.initialize()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()
for cdind, Cd in enumerate(np.arange(cal.cdmin, cal.cdmax, cal.cdinc)):
    simres = cal.update()

    ax.plot(simres[0], simres[1], '-o')  # fit the line
    fig.canvas.draw()  # draw
    time.sleep(0.5)  # sleep

tmp = 0