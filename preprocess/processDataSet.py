# coding = utf-8
import json
import random

source_train_file = "../data/train.json"
source_dev_file = "../data/dev.json"
target_train_file = "../data/rtrain.json"
target_test_file = "../data/rtest.json"
target_dev_file = "../data/rdev.json"

# 打开json文件, 进行遍历
def load_datafile(file_dir):
    with open(file_dir) as table_inf:
        print("Loading data from %s" % file_dir)
        file_data = json.load(table_inf)
    return file_data


def write_datafile(file_dir, data):
    jsonString = json.dumps(data, indent=4)
    with open(file_dir, 'w') as f:
        f.write(jsonString)


def spilitByEx(train_source, dev_source, train_target, dev_target, test_target):
    train = load_datafile(train_source)
    dev = load_datafile(dev_source)

    total = train + dev

    random.shuffle(total)

    train_len = int(len(total) * 0.8)
    dev_len = int(len(total) * 0.1)

    train_list = total[0:train_len]
    dev_list = total[train_len:train_len+dev_len]
    test_list = total[train_len+dev_len:]

    write_datafile(train_target, train_list)
    write_datafile(dev_target, dev_list)
    write_datafile(test_target, test_list)


spilitByEx(source_train_file, source_dev_file, target_train_file, target_dev_file, target_test_file)
