import pandas as pd
import numpy as np

# Block sizes of each digit, half male half female
TRAIN_BLOCK_SIZE = 660
TEST_BLOCK_SIZE = 220

# Relative filepath names
DATA_PATH = 'data/01_raw/'
TEST_FILENAME = 'Test_Arabic_Digit.txt'
TRAIN_FILENAME = 'Train_Arabic_Digit.txt'

INTERMEDIATE_PATH = 'data/02_intermediate/'

# Test raw data read and print
# df_test_lines = pd.read_csv(DATA_PATH + TEST_FILENAME, sep='\n')
# df_test_list = np.split(df_test_lines, df_test_lines[df_test_lines.isnull().all(1)].index)
# print(df_test_list)

# Try to parse data with file opening
empty_lines = 0
digit_num = 0
block_size = TEST_BLOCK_SIZE
line_number = -1

with open(DATA_PATH + TEST_FILENAME, "r") as test_file:
    for line in test_file:

        digit_num = min(empty_lines // block_size, 9)
        line_number += 1

        cur_write_file = f"{INTERMEDIATE_PATH}test_digits/test_0{digit_num}.txt"
        # print(f"empty_lines: {empty_lines}\tcur_write_file: {cur_write_file}")

        # Check if digit is an empty line, this keeps track of digit separation 
        if len(line.strip()) == 0:
            empty_lines += 1

        # do writing to file
        f = open(cur_write_file, "a")
        f.write(line)
        f.close()

empty_lines = 0
digit_num = 0
block_size = TRAIN_BLOCK_SIZE
line_number = -1

with open(DATA_PATH + TRAIN_FILENAME, "r") as test_file:
    for line in test_file:

        digit_num = min(empty_lines // block_size, 9)
        line_number += 1

        cur_write_file = f"{INTERMEDIATE_PATH}train_digits/train_0{digit_num}.txt"
        # print(f"empty_lines: {empty_lines}\tcur_write_file: {cur_write_file}")

        # Check if digit is an empty line, this keeps track of digit separation 
        if len(line.strip()) == 0:
            empty_lines += 1

        # do writing to file
        f = open(cur_write_file, "a")
        f.write(line)
        f.close()