# Block sizes of each digit, half male half female
TRAIN_BLOCK_SIZE = 660
TEST_BLOCK_SIZE = 220

# Relative filepath names (input)
DATA_PATH = 'data/01_raw/'
TEST_FILENAME = 'Test_Arabic_Digit.txt'
TRAIN_FILENAME = 'Train_Arabic_Digit.txt'

# Relative filepath names (output)
INTERMEDIATE_PATH = 'data/02_intermediate/'

# Try to parse data with file opening
empty_lines = 0
digit_num = 0
last_digit = -1
block_size = TEST_BLOCK_SIZE
line_number = -1
cur_write_file = f"{INTERMEDIATE_PATH}test_digits/test_0{digit_num}.txt"
f = open(cur_write_file, "w")

with open(DATA_PATH + TEST_FILENAME, "r") as test_file:
    for line in test_file:

        # Assign current digit/file we are writing to
        digit_num = min(empty_lines // block_size, 9)
        cur_write_file = f"{INTERMEDIATE_PATH}test_digits/test_0{digit_num}.txt"
        if (digit_num != last_digit):
            f.close();
            f = open(cur_write_file, "w")
        last_digit = digit_num

        # Check if digit is an empty line, this keeps track of digit separation 
        if len(line.strip()) == 0:
            empty_lines += 1

        # Write to file
        f.write(line)

empty_lines = 0
digit_num = 0
last_digit = -1
block_size = TRAIN_BLOCK_SIZE
line_number = -1
cur_write_file = f"{INTERMEDIATE_PATH}train_digits/train_0{digit_num}.txt"
f = open(cur_write_file, "w")

with open(DATA_PATH + TRAIN_FILENAME, "r") as test_file:
    for line in test_file:

        # Assign current digit/file we are writing to
        digit_num = min(empty_lines // block_size, 9)
        cur_write_file = f"{INTERMEDIATE_PATH}train_digits/train_0{digit_num}.txt"
        if (digit_num != last_digit):
            f.close();
            f = open(cur_write_file, "w")
        last_digit = digit_num

        # line_number += 1
        # print(f"empty_lines: {empty_lines}\tcur_write_file: {cur_write_file}")

        # Check if digit is an empty line, this keeps track of digit separation 
        if len(line.strip()) == 0:
            empty_lines += 1

        # Write to file
        f.write(line)