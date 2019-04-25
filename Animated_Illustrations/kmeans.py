import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from random import shuffle, random
from matplotlib.ticker import NullLocator
from scipy.spatial import Voronoi


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

x, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6, random_state=0)

#plt.scatter(x[:, 0], x[:, 1], s=50)

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.array(fig.canvas.renderer._renderer)
    
    return buf

seq = []

def find_clusters(x, n_clusters):
    # randomly set cluster centroids
    x_list = list(x)
    shuffle(x_list)
    centroids = np.array([[2 * random(), 4 * random()], [2 * random(), 4 * random()], [2 * random(), 4 * random()]])

    counter = 0

    plt.figure(counter)

    plt.scatter(x[:, 0], x[:, 1], s=50)

    ax = plt.gca()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.xlim(-3.0, 4.0)
    plt.ylim(-1, 6)

    fig1 = plt.gcf()

##    fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
##
##    fig1.savefig('../../Illustrations/kmeans-' + str(counter) + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##    fig1.savefig('../../Illustrations/kmeans-' + str(counter) + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##    fig1.savefig('../../Illustrations/kmeans-' + str(counter) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

    #plt.show()

    
    counter = 1

    while True:

        plt.figure(counter)
        axes = plt.gca()

        # assign labels based on closest centroid
        labels = pairwise_distances_argmin(x, centroids)

        plt.scatter(x[:, 0], x[:, 1], c=[l + 1 for l in labels], s=50, cmap='tab10', zorder=2);

        plt.scatter(centroids[:, 0], centroids[:, 1], c=[1,2,3], s=200, cmap='tab10', marker="s", facecolors='none', zorder=2);
        plt.xlim(-3.0, 4.0)
        plt.ylim(-1, 6)

        vor = Voronoi(centroids)

        # plot
        regions, vertices = voronoi_finite_polygons_2d(vor, 300)
        print("--")
        print(regions)
##        print("--")
##        print(vertices)

        # colorize
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), alpha=0.3, zorder=1)

        ax = plt.gca()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        heading = 'Iteration '+str(counter)
        plt.title(heading)
        plt.tight_layout()
        
        fig1 = plt.gcf()
        nfig = fig2data(fig1)
        seq.append(nfig)
        #ax.set_axis_off()
##        fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
        #plt.margins(0,0)
        #ax.xaxis.set_major_locator(NullLocator())
        #ax.yaxis.set_major_locator(NullLocator())

        
##        fig1.savefig('../../Illustrations/kmeans-' + str(counter) + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##        fig1.savefig('../../Illustrations/kmeans-' + str(counter) + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##        fig1.savefig('../../Illustrations/kmeans-' + str(counter) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

        #plt.show()
        
        # find new centroids as the average of examples
        new_centroids = np.array([x[labels == i].mean(0) for i in range(n_clusters)])
        
        # check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

        counter += 1
    
    return centroids, labels

centroids, labels = find_clusters(x,3)

import os
## Get the directory address of current python file
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_dir) ## Set the current directory as working directory

## The package used to create gif files
import numpngw
numpngw.write_apng('kmeans.png',seq,delay = 500)
