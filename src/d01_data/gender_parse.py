import pandas as pd
import numpy as np

# Block sizes of each digit, half male half female
TRAIN_BLOCK_SIZE = 660
TEST_BLOCK_SIZE = 220

# Relative filepath names
INTERMEDIATE_PATH = 'data/02_intermediate/'
TEST_SUFFIX = 'test_digits/test_0'
TRAIN_SUFFIX = 'train_digits/train_0'
FILE_EXTENSION = '.txt'


block_size = TEST_BLOCK_SIZE

filepath = INTERMEDIATE_PATH + TEST_SUFFIX
for digit in range(10):
    read_file = f"{filepath}{digit}{FILE_EXTENSION}"
    empty_lines = 0

    male_write = f"{INTERMEDIATE_PATH}test_digits/test_0{digit}_male.txt"
    female_write = f"{INTERMEDIATE_PATH}test_digits/test_0{digit}_female.txt"

    f_male = open(male_write, "w")
    f_female = open(female_write, "w")

    with open(read_file, "r") as r_file:
        for line in r_file:

            # Check if digit is an empty line, this keeps track of digit separation 
            if len(line.strip()) == 0:
                empty_lines += 1

            # write first half to male, second to female
            if (empty_lines < block_size / 2):
                f_male.write(line)
            else:
                f_female.write(line)

    # always close file descriptors
    f_male.close()
    f_female.close()


block_size = TRAIN_BLOCK_SIZE

filepath = INTERMEDIATE_PATH + TRAIN_SUFFIX
for digit in range(10):
    read_file = f"{filepath}{digit}{FILE_EXTENSION}"
    empty_lines = 0

    male_write = f"{INTERMEDIATE_PATH}train_digits/train_0{digit}_male.txt"
    female_write = f"{INTERMEDIATE_PATH}train_digits/train_0{digit}_female.txt"

    f_male = open(male_write, "w")
    f_female = open(female_write, "w")

    with open(read_file, "r") as r_file:
        for line in r_file:

            # Check if digit is an empty line, this keeps track of digit separation 
            if len(line.strip()) == 0:
                empty_lines += 1

            # write first half to male, second to female
            if (empty_lines < block_size / 2):
                f_male.write(line)
            else:
                f_female.write(line)

    # always close file descriptors
    f_male.close()
    f_female.close()

