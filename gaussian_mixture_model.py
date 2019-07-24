from __future__ import print_function
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import math

from sklearn.neighbors import KernelDensity

import scipy.integrate as integrate
from sklearn.kernel_ridge import KernelRidge

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

mu1, sigma1 = 3.0, 1.0
mu2, sigma2 = 8.0, 3.5

def sample_points():
    s1 = np.random.normal(mu1, math.sqrt(sigma1), 50)

    s2 = np.random.normal(mu2, math.sqrt(sigma2), 50)

    return list(s1) + list(s2)

def compute_bi(mu1local, sigma1local, mu2local, sigma2local, phi1local, phi2local):
    bis = []
    for xi in x:
        bis.append((sp.stats.norm.pdf(xi, mu1local, math.sqrt(sigma1local)) * phi1local)/(sp.stats.norm.pdf(xi, mu1local, math.sqrt(sigma1local)) * phi1local + sp.stats.norm.pdf(xi, mu2local, math.sqrt(sigma2local)) * phi2local))
    return bis

# generate points used to plot
x_plot = np.linspace(-2, 12, 100)

# generate points and keep a subset of them
x = sample_points()

colors = ['red', 'blue', 'orange', 'green']
lw = 2

mu1_estimate = 1.0
mu2_estimate = 2.0
sigma1_estimate = 1.0
sigma2_estimate = 2.0

phi1_estimate = 0.5
phi2_estimate = 0.5

count = 0
while True:
    plt.figure(count)
    axes = plt.gca()
    axes.set_xlim([-2,12])
    axes.set_ylim([0,0.8])
    plt.xlabel("$x$")
    plt.ylabel("pdf")
    plt.scatter(x, [0.005] * len(x), color='navy', s=30, marker=2, label="training examples")
    plt.plot(x_plot, [sp.stats.norm.pdf(xp, mu1_estimate, math.sqrt(sigma1_estimate)) for xp in x_plot], color=colors[1], linewidth=lw, label="$f(x_i \\mid \\mu_1 ,\\sigma_1^2)$")
    plt.plot(x_plot, [sp.stats.norm.pdf(xp, mu2_estimate, math.sqrt(sigma2_estimate)) for xp in x_plot], color=colors[3], linewidth=lw, label="$f(x_i \\mid \\mu_2 ,\\sigma_2^2)$")
    plt.plot(x_plot, [sp.stats.norm.pdf(xp, mu1, math.sqrt(sigma1)) for xp in x_plot], color=colors[0], label="true pdf")
    plt.plot(x_plot, [sp.stats.norm.pdf(xp, mu2, math.sqrt(sigma2)) for xp in x_plot], color=colors[0])

    plt.legend(loc='upper right')
    plt.tight_layout()

    fig1 = plt.gcf()
    fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
    fig1.savefig('../../Illustrations/gaussian-mixture-model-' + str(count) + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    fig1.savefig('../../Illustrations/gaussian-mixture-model-' + str(count) + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    fig1.savefig('../../Illustrations/gaussian-mixture-model-' + str(count) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    bis1 = compute_bi(mu1_estimate, sigma1_estimate, mu2_estimate, sigma2_estimate, phi1_estimate, phi2_estimate)
    bis2 = compute_bi(mu2_estimate, sigma2_estimate, mu1_estimate, sigma1_estimate, phi2_estimate, phi1_estimate)

    #print bis1[:5]
    #print bis2[:5]

    mu1_estimate = sum([bis1[i] * x[i] for i in range(len(x))]) / sum([bis1[i] for i in range(len(x))])
    mu2_estimate = sum([bis2[i] * x[i] for i in range(len(x))]) / sum([bis2[i] for i in range(len(x))])

    sigma1_estimate = sum([bis1[i] * (x[i] - mu1_estimate)**2 for i in range(len(x))]) / sum([bis1[i] for i in range(len(x))])
    sigma2_estimate = sum([bis2[i] * (x[i] - mu2_estimate)**2 for i in range(len(x))]) / sum([bis2[i] for i in range(len(x))])

    #print mu1_estimate, mu2_estimate
    #print sigma1_estimate, sigma2_estimate

    phi1_estimate = sum([bis1[i] for i in range(len(x))])/float(len(x))
    phi2_estimate = 1.0 - phi1_estimate

    print(phi1_estimate)

    count += 1

    plt.close(count)

    if count > 50:
        break

