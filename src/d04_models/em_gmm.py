import pandas as pd
import numpy as np
import random
import sys
import plot_utils as pu

from datetime import datetime as dt

from sklearn.mixture import GaussianMixture as gm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def parse_labels(matrix, labels, n_clusters):
    """Filter matrix and labels into clustered matrices for scatter plotting"""

    clustered_matrix = []

    for cluster in range(n_clusters):
        filter_arr = create_filter_arr(labels=labels, cluster=cluster)
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
    ax.set_title(f"EM Result for Digit {digit} with {n_clusters} Components")

    plt.show()

def create_filter_arr(labels, cluster):
    """Create boolean flagged filter array to apply to filter np array"""
    filter_arr = []
    for label in labels:
        if label == cluster:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    return filter_arr

class GaussParams:
    """ Gaussian Mixture Model object to encapsulate params """
    def __init__(self, u, pi, cov):
        self.u = u
        self.pi = pi
        self.cov = cov

    def __str__(self):
        return f"u: {self.u}\npi: {self.pi}\ncov: {self.cov}"

# Define all relative file paths to actually get files
DATA_PATH = "data/02_intermediate/"
TRAIN_PATH = f"{DATA_PATH}train_digits/train_0"
TEST_PATH = f"{DATA_PATH}test_digits/test_0"
EXT = ".txt"

# File read params
DIGITS = 10
USE_COEFFS = 7
MODEL_COEFFS = range(USE_COEFFS)
PLOT_COEFFS = [0, 1]
COMPONENTS = 4

# Generate expectation maximization models and results
em_results = []
for digit in range(DIGITS):
    # Read in train file and parse as dataframe
    filename = f"{TRAIN_PATH}{digit}{EXT}"
    df = pd.read_csv(filename, skip_blank_lines=True, delimiter=' ', header=None)
    df.dropna(axis=0, inplace=True)

    # Filter dataframe down to only model coefficient columns
    df_filter = df.iloc[:, MODEL_COEFFS]
    matrix = df_filter.values
    n_clusters = COMPONENTS

    # Apply kmeans on the matrix of values
    gmm = gm(n_components=4)
    gmm.fit(matrix)
    labels = gmm.predict(matrix)
    component_covariances = gmm.covariances_
    component_means = gmm.means_
    component_weights = gmm.weights_

    # Record the GMM results (u, pi, and cov) 
    em = GaussParams(u=component_means, pi=component_weights, cov=component_covariances)
    em_results.append(em)

    # Visualize the kmeans plot as scatter in 2D
    custom_scatter_2D(matrix=matrix, labels=labels, cluster_centers=component_means, n_clusters=n_clusters, digit=digit, coeffs=PLOT_COEFFS)