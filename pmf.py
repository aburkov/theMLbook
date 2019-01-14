from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

x = np.arange(4)
pr = [0.1, 0.3, 0.4, 0.2]

axes = plt.gca()
axes.set_ylim([0,0.6])

plt.bar(x, pr, color="red")
plt.xticks(x, ('1', '2', '3', '4'))
plt.yticks(np.arange(0, 0.7, 0.1))
plt.xlabel("$x$")
plt.ylabel("$pmf$")

fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.12, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/pmf.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/pmf.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/pmf.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)


plt.show()
