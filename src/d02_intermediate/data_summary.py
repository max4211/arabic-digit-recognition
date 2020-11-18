from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import colorbar
import matplotlib.pyplot as plt # to view graphs

"""The goal of this file is to explore differences among digits for normalization 
        e.g. # of blocks, average blocks, variance in blocks, etc."""

DATA_PATH = "data/02_intermediate/"

def file_path(data_type, digit, gender):
    """Generate a file path from data type, digit, and gender"""
    path = f"{DATA_PATH}{data_type}_digits/{data_type}_0{digit}"
    if gender == "male" or gender == "female":
        path = f"{path}_{gender}"
    path = f"{path}.txt"
    return path

PROCESS_DIGITS = 10

def boxplot_digits(data_type, gender):
    fig, ax = plt.subplots()
    ax.set_title(f"Digits and Analysis Frame Spread ({data_type}, {gender})")
    ax.set_ylabel("Number of Analysis Frames")
    ax.set_xlabel("Spoken Digit")
    all_blocks = []

    for digit in range(PROCESS_DIGITS):
        filename = file_path(data_type=data_type, digit=digit, gender=gender)
        blocks = []
        block = 0

        # Open file and build out data
        with open(filename) as file:
            for line in file:
                if len(line.strip()) == 0:
                    if block != 0:
                        blocks.append(block)
                        block = 0
                else:
                    block += 1

        all_blocks.append(blocks)

    ax.boxplot(all_blocks)
    plt.show()

boxplot_digits(data_type="test", gender="female")
boxplot_digits(data_type="test", gender="male")
boxplot_digits(data_type="test", gender="all")
boxplot_digits(data_type="train", gender="female")
boxplot_digits(data_type="train", gender="male")
boxplot_digits(data_type="train", gender="all")