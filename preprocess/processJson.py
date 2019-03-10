# coding = utf-8
import json

source_file = "./data/dev.json"
target_file = "./data/processedDev.json"

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
    for idx, qry in enumerate(file_data):
        # 显示对应的sql语句
        print("The question is:" + qry["question"]+"\n")
        print("The query is:" + qry["query"]+"\n")

        type_of_chart = input("type:\n")
        x_col = input("x_col:\n")
        y_col = input("y_col:\n")

        qry["type_of_chart"] = type_of_chart
        qry["x_col"] = x_col
        qry["y_col"] = y_col

        processed_list.append(qry)
        processed_num += 1

        print(str(len(file_data)-processed_num)+" sqls left\n")

        if idx == 1:
            break
    print("Finished!")
    return processed_list

def write_datafile(file_dir,data):
    jsonString = json.dumps(data, indent=4)
    with open(file_dir, 'w') as f:
        f.write(jsonString)


file_data = load_datafile(source_file)

processed_list = process_json(file_data)

write_datafile(target_file, processed_list)