# coding = UTF-8
import json
import torch
import datetime
import argparse
import numpy as np
from scripts.utils import *
from scripts.model.nlnet.nlnet import NLNet
from backend import database


def get_final_from_seq(seq, db_id, sql, x_col, y_col, type):
    # sql = "select * from Message"
    result = database.select_db(sql, db_id)

    # 0: none
    # 1: bars
    # 2: lines
    type_of_chart = type
    predicted_x_col = x_col
    predicted_y_col = y_col

    cols = result.keys()
    rows_list = []
    xy_list = []
    labels = []
    # 遍历获得数据
    for row in result:
        temp_data_json = {}
        row_json = {}
        for i in range(len(cols)):
            col = cols[i]

            temp = {"label": col}
            if temp not in labels:
                labels.append(temp)

            if i == predicted_x_col:
                temp_data_json["letter"] = row[i]
            elif i == predicted_y_col:
                temp_data_json["frequency"] = row[i]
            row_json[col] = row[i]
        xy_list.append(temp_data_json)
        rows_list.append(row_json)

    sql_result = {}
    sql_result['uflag'] = 0
    sql_result['content'] = ''

    sql_result['type_of_chart'] = type_of_chart
    sql_result['predicted_x_col'] = predicted_x_col
    sql_result['predicted_y_col'] = predicted_y_col
    sql_result["xy_data"] = xy_list
    sql_result['labels'] = labels
    sql_result['table'] = rows_list

    return sql_result


def predict_sql_from_seq(seq, db_id):
    N_word = 300  # 每个词的词维数
    B_word = 42

    FAST = True
    USE_GPU = False
    BATCH_SIZE = 15  # 每个批次处理的输入条目

    # todo:??
    TEST_ENTRY = (True, True, True)  # (AGG, SEL, COND)

    # 加载用户输入
    test_sql_data, test_table_data, schemas = load_test_seq(seq, db_id, FAST=FAST)

    # 加载Word embedding
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), FAST=FAST)

    model = SQLNet(word_emb, N_word=N_word, gpu=USE_GPU)

    # 加载模型参数
    model.sel_pred.load_state_dict(torch.load("saved_models/sel_models.dump", map_location='cpu'))
    model.cond_pred.load_state_dict(torch.load("saved_models/cond_models.dump", map_location='cpu'))
    model.group_pred.load_state_dict(torch.load("saved_models/group_models.dump", map_location='cpu'))
    model.order_pred.load_state_dict(torch.load("saved_models/order_models.dump", map_location='cpu'))

    # 输出到文件
    output = "output.txt"
    print_input_result(model, BATCH_SIZE, test_sql_data, test_table_data, output, schemas, TEST_ENTRY)


def predict_sql(dataset):

    # 每个数的词维数
    N_word=300

    # ??
    B_word=42

    FAST= True
    GPU = False

    # 每个批次处理的输入条目
    BATCH_SIZE = 20

    TEST_ENTRY = (True, True, True)  # (AGG, SEL, COND)

    # 加载json文件
    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, schemas,\
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(dataset, FAST=FAST)

    # 加载预训练的word embedding
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), FAST=FAST)

    model = SQLNet(word_emb, N_word=N_word, gpu=GPU)

    print ("Loading from sel model...")
    model.sel_pred.load_state_dict(torch.load("saved_models/sel_models.dump", map_location='cpu'))
    print ("Loading from sel model...")
    model.cond_pred.load_state_dict(torch.load("saved_models/cond_models.dump", map_location='cpu'))
    print ("Loading from sel model...")
    model.group_pred.load_state_dict(torch.load("saved_models/group_models.dump", map_location='cpu'))
    print ("Loading from sel model...")
    model.order_pred.load_state_dict(torch.load("saved_models/order_models.dump", map_location='cpu'))

    output = "output.txt"
    print_input_result(model, BATCH_SIZE, test_sql_data, test_table_data, output, schemas, TEST_ENTRY)
