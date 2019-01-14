import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})


plt.figure(1)
plt.quiver([0, 0, 0], [0, 0, 0], [2, -2, 1], [3, 5, 0], color=['r','b','g'], angles='xy', scale_units='xy', scale=1)
plt.xlim(-3, 3)
plt.ylim(-1, 6)
plt.xlabel('$x^{(1)}$')
plt.ylabel('$x^{(2)}$')
fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.12, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/vector-0.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/vector-0.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/vector-0.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

plt.figure(2)
plt.scatter([2, -2, 1], [3, 5, 0], color=['r','b','g'])
plt.xlim(-3, 3)
plt.ylim(-1, 6)
plt.xlabel('$x^{(1)}$')
plt.ylabel('$x^{(2)}$')
fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.12, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/vector-1.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/vector-1.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
fig1.savefig('../../Illustrations/vector-1.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0.1)
plt.show()
