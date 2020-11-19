# Relative path of training data
def get_first_dataframe(digit, write_path, read_path):
    # Get the first dataframe from a file
    read_filename = f"{read_path}{digit}.txt"
    write_filename = f"{write_path}{digit}.txt"
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

DATA_PATH = "data/02_intermediate/"

train_read_path = f"{DATA_PATH}train_digits/train_0"
train_write_path = f"{DATA_PATH}single_person/train_0"

test_read_path = f"{DATA_PATH}test_digits/test_0"
test_write_path = f"{DATA_PATH}single_person/test_0"

for digit in range(10):
    get_first_dataframe(digit=digit, write_path=train_write_path, read_path=train_read_path)
    get_first_dataframe(digit=digit, write_path=test_write_path, read_path=test_read_path)
