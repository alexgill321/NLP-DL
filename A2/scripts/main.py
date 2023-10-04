from read_data import parse_file, get_tag_dict
import os
from state import generate_from_data

train_x, train_y = parse_file(os.getcwd() + "/A2/data/train.txt")
dev_x, dev_y = parse_file(os.getcwd() + "/A2/data/dev.txt")
test_x, test_y = parse_file(os.getcwd() + "/A2/data/test.txt")
label_tags = get_tag_dict(os.getcwd() + "/A2/data/tagset.txt")

train_data = generate_from_data(train_x, train_y, label_tags)
dev_data = generate_from_data(dev_x, dev_y, label_tags)
test_data = generate_from_data(test_x, test_y, label_tags)