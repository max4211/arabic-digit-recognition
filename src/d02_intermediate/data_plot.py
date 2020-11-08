import pandas as pd
import numpy as np

# Relative filepath names
INTERMEDIATE_PATH = 'data/02_intermediate/'
FILE_PATH = 'single_person/test_00_female.txt'

# Try to parse data with file opening
empty_lines = 0
digit_num = 0
filename = f"{INTERMEDIATE_PATH}{FILE_PATH}"

# Read file
df = pd.read_csv(filename, delimiter=' ', header=None)
print(df)

# TODO - Print MFCC Coefficients