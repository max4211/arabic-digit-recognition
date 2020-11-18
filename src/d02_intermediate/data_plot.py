# importing dependencies
import pandas as pd # data frame
import numpy as np # matrix math
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from matplotlib import cm
from matplotlib import colors
from matplotlib import colorbar
import matplotlib.pyplot as plt # to view graphs

def plot_mfcc_colorbar(df, filename):
    # Plot MFCC Coefficients
    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(df_01, 0 ,1)
    cmap = cm.coolwarm
    norm = colors.Normalize(df.to_numpy().min(), df.to_numpy().max())
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), ax=ax)
    ax.imshow(mfcc_data, interpolation='nearest', cmap=cmap, origin='lower')
    ax.set_title('MFCC Visualization')
    plt.savefig(filename)
    plt.ylabel(f"MFCC Coefficient")
    plt.xlabel(f"Quefrency (ms)")
    plt.show()

# Relative filepath names
INTERMEDIATE_PATH = 'data/02_intermediate/single_person/test_00_female_'
file_00 = f"{INTERMEDIATE_PATH}00.txt"
file_01 = f"{INTERMEDIATE_PATH}01.txt"

df_00 = pd.read_csv(file_00, delimiter=' ', header=None)
df_01 = pd.read_csv(file_01, delimiter=' ', header=None)

plot_mfcc_colorbar(df=df_00, filename=f"{INTERMEDIATE_PATH}00.png")
plot_mfcc_colorbar(df=df_01, filename=f"{INTERMEDIATE_PATH}01.png")