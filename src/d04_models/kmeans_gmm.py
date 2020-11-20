from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import plot_utils as pu
import sys
from datetime import datetime as dt

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
    ax.set_title(f"K-Means Result for Digit {digit} with {n_clusters} Clusters")

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

def analyze_cluster(matrix, labels, cluster_centers, n_clusters):
    """Compute covariance and pi value for gmm vars of clusters"""
    covariance_matrix = []
    pi_matrix = []

    for cluster in range(n_clusters):
        filter_arr = create_filter_arr(labels=labels, cluster=cluster)
        sub_matrix = matrix[filter_arr]

        pi = len(sub_matrix) / len(matrix)
        covariance = np.cov(np.transpose(sub_matrix))

        pi_matrix.append(pi)
        covariance_matrix.append(covariance)
        
    return (covariance_matrix, pi_matrix)

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
CLUSTERS = 4

# Generate gaussian models and results
gauss_results = []
for digit in range(DIGITS):
    # Read in train file and parse as dataframe
    filename = f"{TRAIN_PATH}{digit}{EXT}"
    df = pd.read_csv(filename, skip_blank_lines=True, delimiter=' ', header=None)
    df.dropna(axis=0, inplace=True)

    # Filter dataframe down to only model coefficient columns
    df_filter = df.iloc[:, MODEL_COEFFS]
    matrix = df_filter.values
    n_clusters = CLUSTERS

    # Apply kmeans on the matrix of values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Record the GMM results (u, pi, and cov)
    cluster_covariance, cluster_pi = analyze_cluster(matrix=matrix, labels=labels, cluster_centers=cluster_centers, n_clusters=n_clusters)    
    gauss = GaussParams(u=cluster_centers, pi=cluster_pi, cov=cluster_covariance)
    gauss_results.append(gauss)

    # Visualize the kmeans plot as scatter in 2D
    # custom_scatter_2D(matrix=matrix, labels=labels, cluster_centers=cluster_centers, n_clusters=n_clusters, digit=digit, coeffs=PLOT_COEFFS)

def classify_dataframe(df, gauss_results, debug):
    """classify a dataframe based on gaussian results"""
    # Perform classification on some test data
    posterior_all = []

    for d in range(DIGITS):
        """Iterate over all digits (possible classifications"""
        posterior_digit = 1

        for n, row in df.iterrows():
            """Iterate over all n frames of the sample"""
            frames_n = row.to_numpy()[MODEL_COEFFS]

            sum_m = 0
            result_m = gauss_results[d]
            """Iterate over all results from gmm parameters"""
            cov, pi, u = result_m.cov, result_m.pi, result_m.u   

            for m in range(len(u)):
                """Iterate over all m dimensions of mixture model"""
                u_m = u[m]
                cov_m = cov[m]
                pi_m = pi[m]

                y = multivariate_normal.logpdf(x=frames_n, mean=u_m, cov=cov_m)
                posterior_i = y * pi_m
                sum_m += posterior_i

            # end sum over all gauss components for digit
            posterior_digit *= sum_m
            
            # circuit break on underflow, no longer needed with logpdf
            # y = multivariate_normal.pdf(x=frames_n, mean=u_m, cov=cov_m)  # this causes underflow
            if posterior_digit == 0:
                sys.exit()

        # TODO - normalize by the number of samples (is this necessary?)
        # end product of all n frames
        if (debug):
            print(f"digit: {d}\tposterior_digit: {posterior_digit}")
        posterior_all.append(posterior_digit)
    
    classification = posterior_all.index(max(posterior_all))
    return (classification, posterior_all)

def get_all_dataframes(digit, write_path, read_path, stopwatch):
    """
    Get all of the dataframes for a single digit 
    Use single_person data folder as intermediary for pandas read csv ease of use
    """
    start_time = dt.now()
    read_filename = f"{read_path}{digit}.txt"
    write_filename = f"{write_path}{digit}.txt"

    f = open(write_filename, "w")
    line_count = 0

    df_all = []

    # Open file and build out data
    with open(read_filename, "r") as file:
        for line in file:
            if len(line.strip()) != 0:
                f.write(line)
                line_count += 1
            elif line_count > 0:
                # Close file descriptor, read in written data, update dataframes
                f.close()
                df = pd.read_csv(write_filename, skip_blank_lines=True, delimiter=' ', header=None)
                df_all.append(df)

                # Reset line count and file descriptor for new dataframe parse
                line_count = 0
                f = open(write_filename, "w")

    # Likely have one more (no missing line on final line)
    if line_count > 0:
        f.close()
        df = pd.read_csv(write_filename, skip_blank_lines=True, delimiter=' ', header=None)
        df_all.append(df)

    end_time = dt.now()
    total_time = (end_time - start_time).total_seconds()

    if (stopwatch):
        print(f"Parsed {len(df_all)} frames in {total_time} sec")

    return df_all
     
def print_summary(digit, total_time, correct, utterances):
    """Output summary from classification to console"""
    accuracy = correct / utterances * 100
    accuracy = round(accuracy, 3)
    dt_format = "%H:%M:%S"
    cur_time = dt.strftime(dt.now(), dt_format) 
    print(f"#{digit}\taccuracy: {accuracy}%\tcorrect: {correct}\tutterances: {utterances}\ttotal_time: {round(total_time, 3)} sec\tcur_time: {cur_time}")

DATA_PATH = "data/02_intermediate/"
test_read_path = f"{DATA_PATH}test_digits/test_0"
test_write_path = f"{DATA_PATH}single_person/test_0"


classify_every = 10
classify_results = []
for digit in range(DIGITS):
    total_classified = 0
    correct = 0
    df_all = get_all_dataframes(digit=digit, write_path=test_write_path, read_path=test_read_path, stopwatch=False)
    classify_digit = [0]*DIGITS

    index = 0
    start_time = dt.now()
    for df in df_all:
        if index % classify_every == 0:
            (classification, posterior_all) = classify_dataframe(df=df, gauss_results=gauss_results, debug=False)    
            total_classified += 1
            classify_digit[classification] += 1
            if classification == digit:
                correct += 1
        index += 1

    classify_results.append(classify_digit)

    end_time = dt.now()
    total_time = (end_time - start_time).total_seconds()
    print_summary(digit=digit, total_time=total_time, correct=correct, utterances=total_classified)

# end
print(f"classify_results:")
# df_results = pd.DataFrame(classify_results)
# df_results

for result in classify_results:
    print(result)