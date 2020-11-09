# importing dependencies
import pandas as pd # data frame
import numpy as np # matrix math
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from matplotlib import cm
import matplotlib.pyplot as plt # to view graphs

# Relative filepath names
INTERMEDIATE_PATH = 'data/02_intermediate/single_person/test_00_female_'
file_00 = f"{INTERMEDIATE_PATH}00.txt"
file_01 = f"{INTERMEDIATE_PATH}01.txt"

print(f"file_00: {file_00}")
df_00 = pd.read_csv(file_00, delimiter=' ', header=None)



# Plot MFCC Coefficients
fig, ax = plt.subplots()
mfcc_data= np.swapaxes(df_00, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('MFCC (00)')
plt.savefig(f"{INTERMEDIATE_PATH}00.png")
plt.show()


print(f"file_01: {file_01}")
df_01 = pd.read_csv(file_01, delimiter=' ', header=None)

fig, ax = plt.subplots()
mfcc_data= np.swapaxes(df_01, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('MFCC (01)')
plt.savefig(f"{INTERMEDIATE_PATH}01.png")
plt.show()