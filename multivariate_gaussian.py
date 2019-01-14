import numpy as np
import matplotlib

from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 4/5], [3/4, 2]]  # diagonal covariance

x, y = np.random.multivariate_normal(mean, cov, 200).T
fig = plt.figure(1)
plt.plot(x, y, 'o')
plt.axis('equal')
plt.xlabel('$x^{(1)}$')
plt.ylabel('$x^{(2)}$')

fig.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.02, hspace = 0, wspace = 0)
fig.savefig('../../Illustrations/multivariate-gaussian-0.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig.savefig('../../Illustrations/multivariate-gaussian-0.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig.savefig('../../Illustrations/multivariate-gaussian-0.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)

fig1 = plt.figure(2)

ax = Axes3D(fig1)

x1, y1 = np.mgrid[-5:5:.2, -5:5:.2]
pos = np.empty(x1.shape + (2,))
pos[:, :, 0] = x1; pos[:, :, 1] = y1
rv = multivariate_normal(mean, cov)
#ax.plot_surface(x1, y1, rv.pdf(pos), rstride=1, cstride=1, alpha=0.8, cmap='viridis', edgecolor='none')
ax.plot_wireframe(x1, y1, rv.pdf(pos), rstride=2, cstride=2, color='gray')

z = [0] * len(x)
ax.scatter(x, y, z)

ax.set_xlabel('$x^{(1)}$')
ax.set_ylabel('$x^{(2)}$')
ax.set_zlabel('pdf');
ax.set_zticks([])
ax.set_xticks([])
ax.set_yticks([])

#ax.view_init(14, -77)

fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.9, left = 0.08, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/multivariate-gaussian-1.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/multivariate-gaussian-1.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/multivariate-gaussian-1.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

#plt.show()
