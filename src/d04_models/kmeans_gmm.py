from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import plot_utils as pu

# Define all relative file paths to actually get files
DATA_PATH = "data/02_intermediate/"
TRAIN_PATH = f"{DATA_PATH}train_digits/train_0"
TEST_PATH = f"{DATA_PATH}test_digits/test_0"
EXT = ".txt"

# TEST KMEANS ON SINGLE DIGIT
# File read params
digit = 0
filename = f"{TRAIN_PATH}{digit}{EXT}"
df = pd.read_csv(filename, skip_blank_lines=True, delimiter=' ', header=None)
df.dropna(axis=0, inplace=True)

# kmeans params
matrix = df.values
n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(matrix)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

print(f"labels: {labels}")
print(f"cluster_centers: {cluster_centers}")

# scatter-3D plots (manually do this)
def custom_scatter_3D(matrix, labels, cluster_centers, coeffs=[0, 1, 2]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = matrix[:, coeffs[0]]
    ys = matrix[:, coeffs[1]]
    zs = matrix[:, coeffs[2]]

    print(f"xs: {xs}")
    print(f"ys: {ys}")
    print(f"zs: {zs}")

    ax.scatter(xs, ys, zs)
    ax.set_xlabel(f"MFCC {coeffs[0]}")
    ax.set_ylabel(f"MFCC {coeffs[1]}")
    ax.set_zlabel(f"MFCC {coeffs[2]}")

    plt.show()

def parse_labels(matrix, labels, n_clusters):
    """Filter matrix and labels into clustered matrices for scatter plotting"""

    clustered_matrix = []

    for cluster in range(n_clusters):
        filter_arr = []
        for label in labels:
            if label == cluster:
                filter_arr.append(True)
            else:
                filter_arr.append(False)
            
        sub_matrix = matrix[filter_arr]
        clustered_matrix.append(sub_matrix)

    return clustered_matrix

def custom_scatter_2D(matrix, labels, cluster_centers, n_clusters, digit, coeffs=[0, 1]):
    """Scatter plot of 2 dimensions of kmeans results"""
    fig = plt.figure()
    ax = fig.add_subplot()

    clustered_matrix = parse_labels(matrix=matrix, labels=labels, n_clusters=n_clusters)
    # for cluster in clustered_matrix:
    for index in range(len(clustered_matrix)):
        cluster = clustered_matrix[index]
        xs = cluster[:, coeffs[0]]
        ys = cluster[:, coeffs[1]]
        color = pu.random_color()
        ax.scatter(x=xs, y=ys, s=0.5, color=color, marker=pu.random_marker())
    
    ax.set_xlabel(f"MFCC {coeffs[0]}")
    ax.set_ylabel(f"MFCC {coeffs[1]}")
    ax.set_title(f"K-Means Result for Digit {digit} with {n_clusters} Clusters")

    plt.show()

custom_scatter_2D(matrix=matrix, labels=labels, cluster_centers=cluster_centers, n_clusters=n_clusters, digit=digit, coeffs=[0, 1])
# custom_scatter_3D(matrix=matrix, labels=labels, cluster_centers=cluster_centers, coeffs=[0, 1, 2])