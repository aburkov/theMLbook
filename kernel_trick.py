import random
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

def f_outer(x1):
	result = []
	for x in x1:
		side = random.uniform(0, 1)
		sq = math.sqrt(10 * 10 - x * x)
		if side > 0.5:
			sq = sq * (-1)
		result.append(sq)
	return np.asarray(result)

def f_inner(x1):
	result = []
	for x in x1:
		side = random.uniform(0, 1)
		sq = math.sqrt(3 * 3 - x * x)
		if side > 0.5:
			sq = sq * (-1)
		result.append(sq)
	return np.asarray(result)


# generate points and keep a subset of them
x_inner = np.linspace(-3, 3, 100)
x_outer = np.linspace(-10, 10, 100)

rng = np.random.RandomState(0)
rng.shuffle(x_inner)
rng.shuffle(x_outer)

x_inner = np.sort(x_inner[:30])
x_outer = np.sort(x_outer[:30])

noize = [(-1 + np.random.random()) for i in range(len(x_inner))]
y_inner = f_inner(x_inner) + noize

noize = [(-1 + np.random.random()) for i in range(len(x_outer))]
y_outer = f_outer(x_outer) + noize

colors = ['blue', 'red']#, 'orange'
lw = 2

type_of_regression = ["linear regression", "regression of degree 10"]
fit = ["fit", "overfit"]

plt.figure(1)
axes = plt.gca()
axes.set_xlim([-11,11])
axes.set_ylim([-11,11])

plt.scatter(x_inner, y_inner, color='navy', s=30, marker='o')
plt.scatter(x_outer, y_outer, color='red', s=30, marker='o')

fig1 = plt.gcf()

fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/kernel-trick-0.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/kernel-trick-0.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/kernel-trick-0.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

x_inner_transformed = np.asarray([x * x for x in x_inner])
y_inner_transformed = np.asarray([math.sqrt(2) * x * y for x, y in zip(x_inner, y_inner)])
z_inner_transformed = np.asarray([y * y for y in y_inner])

x_outer_transformed = np.asarray([x * x for x in x_outer])
y_outer_transformed = np.asarray([math.sqrt(2) * x * y for x, y in zip(x_outer, y_outer)])
z_outer_transformed = np.asarray([y * y for y in y_outer])

fig = plt.figure(2)
ax = Axes3D(fig)
ax.set_yticks([-75, 0, 75])
#ax.set_xlim([-10,120])
#$ax.set_ylim([-120,120])
#ax.set_zlim([-120,120])

ax.scatter(x_inner_transformed, y_inner_transformed, z_inner_transformed, color='navy', marker='o')
ax.scatter(x_outer_transformed, y_outer_transformed, z_outer_transformed, color='red', marker='o')

ax.view_init(14, -77)

fig.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
fig.savefig('../../Illustrations/kernel-trick-1.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig.savefig('../../Illustrations/kernel-trick-1.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig.savefig('../../Illustrations/kernel-trick-1.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

#plt.show()
