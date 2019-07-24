from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import math

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from random import shuffle
from scipy.spatial import Voronoi
from scipy.spatial import distance

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 25})

random_state = 0

## how many clusters do you want in your synthetic data?
centers = 2

x, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.6, random_state=random_state)

plt.figure(10000)
plt.scatter(x[:, 0], x[:, 1], s=20, cmap='viridis');
plt.xlim(-1, 4.0)
plt.ylim(-1, math.ceil(max(x[:, 1])))
plt.xticks(np.arange(int(min(x[:, 0])), math.ceil(max(x[:, 0]))+1, 1))
plt.yticks(np.arange(int(min(x[:, 1])), math.ceil(max(x[:, 1]))+1, 2), rotation='vertical')

ax = plt.gca()
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.16, right = 0.98, left = 0.12, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '.eps', format='eps', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '.pdf', format='pdf', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '.png', dpi=1000)

x_list = list(x)

random.Random(random_state).shuffle(x_list)

x_split = {}

x_split["train"] = np.array(x_list[:len(x_list)/2])

x_split["test"] = np.array(x_list[len(x_list)/2:])

centroids_splits = {}
labels_splits = {}
counter = 100

def find_clusters(x, n_clusters, current_split):

    current_split_suffled = list(x_split[current_split])[:]
    shuffle(current_split_suffled)
    current_split_suffled = np.array(current_split_suffled)

    centroids = np.array(current_split_suffled[:n_clusters])

    while True:

        # assign labels based on closest centroid
        #print centroids

        #print "len train", len(x_split[current_split])
        labels = pairwise_distances_argmin(x_split[current_split], centroids)
        #print "len labels", len(labels)

        
        # find new centroids as the average of examples
        new_centroids = np.array([x_split[current_split][labels == i].mean(0) for i in range(n_clusters)])
        
        # check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

def get_examples_from_cluster(j, test_points, test_labels):
    examples = []
    for e, l in zip(test_points, test_labels):
        if l == j:
            examples.append(e)
    return examples

def get_closest_centroid(example, centroids):
    min_distance = sys.float_info.max
    min_centroid = 0
    for c in centroids:
        if distance.euclidean(example, c) < min_distance:
            min_distance = distance.euclidean(example, c)
            min_centroid = c
    return min_centroid

def compute_strength(k, train_centroids, test_points, test_labels):
    D = np.zeros(shape=(len(test_points),len(test_points)))
    for x1, l1, c1 in zip(test_points, test_labels, list(range(len(test_points)))):
        for x2, l2, c2 in zip(test_points, test_labels, list(range(len(test_points)))):
            if tuple(x1) != tuple(x2):
                if tuple(get_closest_centroid(x1, train_centroids)) == tuple(get_closest_centroid(x2, train_centroids)):
                    D[c1,c2] = 1.0

    ss = []
    for j in range(k):
        s = 0
        examples_j = get_examples_from_cluster(j, test_points, test_labels)
        for x1, l1, c1 in zip(test_points, test_labels, list(range(len(test_points)))):
            for x2, l2, c2 in zip(test_points, test_labels, list(range(len(test_points)))):
                if tuple(x1) != tuple(x2) and l1 == l2 and l1 == j:
                    s += D[c1,c2]
        s = (1.0/(float(len(examples_j))*float(len(examples_j) - 1)))*s
        ss += [s]

    return min(ss)

strengths = []
ks = [1,2,3,4,5,6,7,8]
for k in ks:
    print("k", k)
    for current_split in ["train", "test"]:
        counter += 1
        centroids, labels = find_clusters(x, k, current_split)

        centroids_splits[current_split] = centroids
        labels_splits[current_split] = labels
    s = compute_strength(k, centroids_splits["train"], x_split["test"], labels_splits["test"])
    strengths += [s]
    print(s)

plt.figure(10001)
plt.plot(ks, strengths);
plt.xticks(np.arange(1, 9, 1))
plt.yticks(np.arange(0, 1.05, 0.2), rotation='vertical')

ax = plt.gca()
ax.set_xlabel('$k$')
ax.set_ylabel('$\\operatorname{ps}(k)$')

fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.15, right = 0.98, left = 0.15, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '_search.eps', format='eps', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '_search.pdf', format='pdf', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '_search.png', dpi=1000)
