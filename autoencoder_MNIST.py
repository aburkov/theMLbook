import pylab as plt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

from keras.datasets import mnist

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 25})

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

m = Sequential()
m.add(Dense(512,  activation='elu', input_shape=(784,)))
m.add(Dense(128,  activation='elu'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(x_train, x_train, batch_size=128, epochs=5, verbose=1, 
                validation_data=(x_test, x_test))

encoder = Model(m.input, m.get_layer('bottleneck').output)
embedding = encoder.predict(x_train)  # bottleneck representation

plt.figure()

plt.scatter(embedding[:,0], embedding[:,1], c=y_train, s=0.1, cmap='Spectral')

plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

ax = plt.gca()
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

fig1 = plt.gcf()

fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/autoencoder-MNIST.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/autoencoder-MNIST.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
fig1.savefig('../../Illustrations/autoencoder-MNIST.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

plt.show()
