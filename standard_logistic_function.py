import matplotlib.pylab as plt
import matplotlib
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

def sigmoid(x):
    """
    evaluate the boltzman function with midpoint xmid and time constant tau
    over x
    """
    return 1. / (1. + np.exp(-x))


x = np.linspace(-6, 6, 100)
S = sigmoid(x)
plt.plot(x, S, color='red', lw=2)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")

fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/standard_logistic_function.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/standard_logistic_function.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/standard_logistic_function.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

plt.show()
