import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 25})

def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * (x)


# generate points used to plot
x_plot = np.linspace(-5, 2, 100)

# generate points and keep a subset of them
x = np.linspace(-5, 2, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:50])
noize = [(-5 + np.random.random()*5) for i in range(len(x))]
y = f(x) + noize

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['red', 'blue', 'orange']
lw = 2

def kernel(x1, x2, b = 2):
    z = (x1 - x2) / b
    return (1/math.sqrt(2 * 3.14)) * np.exp(-z**2/2)

fit = ["fit", "small overfit", "big overfit"]
for count, degree in enumerate([0.1, 0.5, 3]):
    plt.figure(count)
    axes = plt.gca()
    axes.set_xlim([-5,2])
    axes.set_ylim([-10,30])
    plt.scatter(x, y, color='navy', s=30, marker='o', label="training examples")
    model = KernelRidge(alpha=0.01, kernel=kernel, kernel_params = {'b':degree})
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="b = " + str(degree))

    plt.legend(loc='upper right')
    fig1 = plt.gcf()
    fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
    fig1.savefig('../../Illustrations/kernel-regression-' + str(count) + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    fig1.savefig('../../Illustrations/kernel-regression-' + str(count) + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    fig1.savefig('../../Illustrations/kernel-regression-' + str(count) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)


plt.show()
