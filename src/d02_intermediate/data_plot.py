# importing dependencies
import random
import pandas as pd # data frame
import numpy as np # matrix math
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import colorbar
import matplotlib.pyplot as plt # to view graphs
from itertools import combinations # all combinations for mfcc plot

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

def all_colors():
    return ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

def random_color():
    pallette = all_colors()
    return cc(random.choice(pallette))

def all_lines():
    return ["-", "--", "-.", "."]

def all_markers():
    return [".", ",", "o", "v", "^", "1", "8", "*", "H", "d"]

def random_marker():
    return random.choice(all_markers())

def random_rgb():
    return (random.random(), random.random(), random.random())

def random_line_style():
    color = random_rgb()
    line = random.choice(all_lines())
    line_style = f"{color}{line}"
    print(f"line_style: {line_style}")
    return line_style

def plot_mfcc_colorbar(df, filename):
    # Plot MFCC Coefficients
    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(df_01, 0 ,1)
    cmap = cm.coolwarm
    norm = mcolors.Normalize(df.to_numpy().min(), df.to_numpy().max())
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), ax=ax)
    ax.imshow(mfcc_data, interpolation='nearest', cmap=cmap, origin='lower')
    ax.set_title('MFCC Visualization')
    plt.ylabel(f"MFCC Coefficient")
    plt.xlabel(f"Quefrency (ms)")
    plt.savefig(filename)
    plt.show()

def plot_mfcc_linear(df, coeffs, filename, save):
    # Plot MFCC Coefficients as lines and analysis frame
    fig, ax = plt.subplots()
    for i in range(coeffs):
        plt.plot(df[i], color=random_color(), marker=random_marker(), markersize=4, label=f"MFCC {i+1}")
    ax.set_title('Analysis Frame vs. Coefficient Magnitude')
    ncols = 1 if (coeffs <= 5) else coeffs // 3
    plt.legend(ncol=ncols)
    plt.ylabel(f"Magnitude of MFCC")
    plt.xlabel(f"Quefrency (ms)")
    if (save):
        plt.savefig(filename)
    plt.show()

def plot_mfcc_i_j(df, digit, i, j):
    # Plot MFCC Coefficient i vs coefficient j
    fig, ax = plt.subplots()
    marker_size = 0.5
    plt.scatter(x=df[i], y=df[j], s=marker_size, marker=random_marker())        
    ax.set_title(f"MFCC {i} vs {j} for digit {digit}")
    plt.ylabel(f"MFCC {j}")
    plt.xlabel(f"MFCC {i}")
    plt.show()

# Relative filepath names
INTERMEDIATE_PATH = 'data/02_intermediate/single_person/test_00_female_'
file_00 = f"{INTERMEDIATE_PATH}00.txt"
file_01 = f"{INTERMEDIATE_PATH}01.txt"

# Load file data into dataframes
df_00 = pd.read_csv(file_00, delimiter=' ', header=None)
df_01 = pd.read_csv(file_01, delimiter=' ', header=None)

# Construct mfcc colorbar from dataframes
# plot_mfcc_colorbar(df=df_00, filename=f"{INTERMEDIATE_PATH}colorbar_00.png")
# plot_mfcc_colorbar(df=df_01, filename=f"{INTERMEDIATE_PATH}colorbar_01.png")

# Plot MFCC Coefficients as function of analysis window
COEFFS = 4
# plot_mfcc_linear(df=df_00, coeffs=COEFFS, filename=f"{INTERMEDIATE_PATH}linear_00.png", save=True)
# plot_mfcc_linear(df=df_01, coeffs=COEFFS, filename=f"{INTERMEDIATE_PATH}linear_01.png", save=True)

# Generate all two dimensional combinatoins of MFCC coefficients
comb = combinations(range(0, 13), 2)
DIGIT = 0

total_plots = 20

# Relative filepath names
INTERMEDIATE_PATH = 'data/02_intermediate/train_digits/train_'
test_file = f"{INTERMEDIATE_PATH}00.txt"

# Load file data into dataframes
df_test = pd.read_csv(test_file, delimiter=' ', header=None)

# Plot MFCC Coefficients against eachother for i, i+1
# for i in range(12):
#     plot_mfcc_i_j(df=df_test, digit=DIGIT, i=i, j=i+1)

# Plot MFCC Coefficients against eachother for ALL pairwise combos
# for combo in list(comb):
#     if (total_plots <= 0):
#         break
#     i = combo[0]
#     j = combo[1]
#     plot_mfcc_i_j(df=df_test, digit=DIGIT, i=i, j=j)
#     total_plots -= 1;

# Relative path of training data
TRAIN_PATH = 'data/02_intermediate/train_digits/train_0'
WRITE_PATH = 'data/02_intermediate/single_person/train_0'

def get_first_dataframe(digit):
    """Get the first dataframe from a file"""
    read_filename = f"{TRAIN_PATH}{digit}.txt"
    write_filename = f"{WRITE_PATH}{digit}.txt"

    f = open(write_filename, "w")
    line_count = 0
    # Open file and build out data
    with open(read_filename, "r") as file:
        for line in file:
            if len(line.strip()) != 0:
                f.write(line)
                line_count += 1
            elif line_count > 0:
                f.close()
                break

    return pd.read_csv(write_filename, delimiter=' ', header=None)

def plot_mfcc_linear_colorbar(df, digit, coeffs):
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(1, 2, 1)
    mfcc_data= np.swapaxes(df_01, 0 ,1)
    cmap = cm.coolwarm
    norm = mcolors.Normalize(df.to_numpy().min(), df.to_numpy().max())
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), ax=ax)
    ax.imshow(mfcc_data, interpolation='nearest', cmap=cmap, origin='lower')
    ax.set_title(f"MFCC Visualization (Digit {digit})")
    ax.set_ylabel(f"MFCC Coefficient")
    ax.set_xlabel(f"Quefrency (ms)")

    ax = fig.add_subplot(1, 2, 2)
    # for i in range(coeffs):
    #     print(f"i: {i}")
    #     df = df[i]
    #     size = df.shape[0]
    #     x = np.arange(size) # range(size)
    #     print(f"df: \n{df}")
    #     print(f"x: {x}")
    #     ax.scatter(x=x, y=df, color=random_color(), marker=random_marker(), s=1, label=f"MFCC {i+1}")

    for i in range(coeffs):
        ax.plot(df[i], color=random_color(), marker=random_marker(), markersize=4, label=f"MFCC {i+1}")
    ax.set_title(f"Frame vs. MFCC Magnitude (Digit {digit})")
    ncols = 1 if (coeffs <= 5) else coeffs // 3
    plt.legend(ncol=ncols)
    ax.set_ylabel(f"Magnitude of MFCC")
    ax.set_xlabel(f"Quefrency (ms)")

    plt.show()

# Get first dataframe for all digits and make subplots of mfcc and visualization
COEFFS = 6
for digit in range(10):
    df = get_first_dataframe(digit=digit)
    plot_mfcc_linear_colorbar(df=df, digit=digit, coeffs=COEFFS)

# Over all digits, get dataframes and then plot mfcc linear and other in subplots

def subplot_mfccs(digits, max_mfcc, fig_width, fig_height):
    """Create subplots of multiple digits"""
    dataframes = []
    for digit in digits:
        filename = f"{TRAIN_PATH}{digit}.txt"
        df = pd.read_csv(filename, delimiter=' ', header=None)
        dataframes.append(df)

    for row in range(max_mfcc):
        fig = plt.figure(figsize=(fig_width, fig_height))
        for col in range(len(digits)):
            # Fetch digit and dataframe
            digit = digits[col]
            df = dataframes[col]
            i, j = row, row+1

            # Assign plot subplots and axes
            marker_size = 0.5
            ax = fig.add_subplot(1, len(digits), col+1)
            ax.set_title(f"MFCC {i} vs {j} for digit {digit}")
            
            plt.scatter(x=df[i], y=df[j], s=marker_size, marker=random_marker(), color=random_color())        
            ax.set_title(f"Digit {digit}")
            plt.xlabel(f"MFCC {i}")

            if (col == 0):
                plt.ylabel(f"MFCC {j}")
        
        plt.show()

digits = [0, 4, 9]
max_mfcc = 4 #12
fig_width, fig_height = 14, 4
# subplot_mfccs(digits=digits, max_mfcc=max_mfcc, fig_width=fig_width, fig_height=fig_height)