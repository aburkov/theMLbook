import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

from sklearn.kernel_ridge import KernelRidge

mu1, sigma1 = 3, 0.4
mu2, sigma2 = 5, 0.6

def sample_points():
    s1 = np.random.normal(mu1, sigma1, 20)

    s2 = np.random.normal(mu2, sigma2, 20)

    return list(s1) + list(s2)

# generate points used to plot
x_plot = np.linspace(0, 8, 100)

# generate points and keep a subset of them
x = sample_points()

lw = 2

def kernel(x1, x2, b = 2):
    z = (x1 - x2) / b
    return (1/math.sqrt(2 * 3.14)) * np.exp(-z**2/2)

def fb(x, data, b):
    return 1/(len(data)*b) * sum([kernel(x, xi, b) for xi in data])

def sum_pdf(x):
    result = []
    for i in range(len(x)):
        result.append((sp.stats.norm.pdf(x, mu1, sigma1)[i] + sp.stats.norm.pdf(x, mu2, sigma2)[i])/2)
    return result

plt.figure(0)
axes = plt.gca()
axes.set_ylim([0,0.6])
plt.plot(x_plot,sum_pdf(x_plot), color='red')
section = np.arange(0, 8, 1/20.)
plt.fill_between(section,sum_pdf(section), color='#e6eeff')
plt.text(3.2, 0.04, "Area = 1.0", fontsize=18)
plt.xlabel("$x$")
plt.ylabel("$pdf$")

#plt.legend(loc='lower left')
fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.12, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/pdf.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/pdf.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/pdf.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)

plt.show()
