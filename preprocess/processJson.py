# coding = utf-8
import json

source_file = "../data/processed/temp/train.json"
target_file = "./data/train.json"

# 打开json文件, 进行遍历
def load_datafile(file_dir):
    with open(file_dir) as table_inf:
        print("Loading data from %s" % file_dir)
        file_data = json.load(table_inf)

    total_num = 0
    del_num = 0
    # 过滤sel_num < 2 的json
    for one_sql in file_data[:]:
        sel_num = len(one_sql["sql"]["select"][1])
        if sel_num < 2:
            del_num += 1
            file_data.remove(one_sql)
        else:
            total_num += 1
    return file_data



def process_json(file_data):
    processed_num = 0
    processed_list = []
    qry_before = None
    for idx, qry in enumerate(file_data):
        # 显示对应的sql语句
        print("The question is:" + qry["question"] + "\n")
        print("The query is:" + qry["query"] + "\n")
        if qry_before != None and qry['query'] == qry_before['query']:
            print("Same query!\n")
            qry['type_of_chart'] = qry_before['type_of_chart']
            qry["x_col"] = qry_before['x_col']
            qry["y_col"] = qry_before['y_col']
        else:
            type_of_chart = input("type:")
            if type_of_chart != '1' and type_of_chart != '2':
                x_col = 0
                y_col = 0
            else:
                x_col = input("x_col:")
                y_col = input("y_col:")
            qry["type_of_chart"] = type_of_chart
            qry["x_col"] = x_col
            qry["y_col"] = y_col
        qry_before = qry
        processed_list.append(qry)
        processed_num += 1

        print(str(len(file_data)-processed_num)+" sqls left\n")

    print("Finished!")
    return processed_list


def split_json(file_data):
    unprocessed_list = []
    processed_list = []
    for idx, qry in enumerate(file_data):
        if int(qry['type_of_chart']) > 2:
            unprocessed_list.append(qry)
        else:
            processed_list.append(qry)
    return unprocessed_list, processed_list


def write_datafile(file_dir,data):
    jsonString = json.dumps(data, indent=4)
    with open(file_dir, 'w') as f:
        f.write(jsonString)


def count_type(file_data):
    type0 = 0
    type1 = 0
    type2 = 0

    for idx, qry in enumerate(file_data):
        if int(qry['type_of_chart']) == 0:
            type0 += 1
        elif int(qry['type_of_chart']) == 1:
            type1 += 1
        else:
            type2 += 1
    return (type0, type1, type2)


def slipt_data():
    file_data = load_datafile(source_file)

    # processed_list = process_json(file_data)
    unprocessed_list, processed_list = split_json(file_data)

    unprocessed_file = "../data/unprocessed_dev.json"
    # unprocessed
    write_datafile(unprocessed_file, unprocessed_list)

    processed_file = "../data/processed_dev.json"
    # processed
    write_datafile(processed_file, processed_list)

    print("Finished!")

file_data = load_datafile(source_file)

a, b, c = count_type(file_data)

print(a, b, c)
