# -*- coding: UTF-8 -*-
import json
import torch
import datetime
import argparse
import numpy as np
from scripts.utils import *
from scripts.model.nlnet.nlnet import NLNet
from scripts.model.chartnet.chartnet import chartNet, AGG_OPS
from backend import database


# 封装成前段所需的格式
def get_final_from_seq(seq, db_id):
    # sql = "select * from Message"
    sql = "SELECT count(*) FROM head WHERE age  >  56"
    '''
    x_col = 1
    y_col = 5
    types = 1
    '''
    # sql = predict_one_sql(seq, db_id)
    db_id = "department_management"
    result = database.select_db(sql, db_id)
    cols = result.keys()

    type_of_chart = predicted_x_col = predicted_y_col = 0

    chart = False
    if len(cols) > 1:
        chart = True
        one_chart = predict_one_chart(sql, cols)
        type_of_chart = int(one_chart['type'])
        predicted_x_col = int(one_chart['x_col'])
        predicted_y_col = int(one_chart['y_col'])

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

            if chart:
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


def predict_one_sql(seq, db_id):
    N_word = 300
    B_word = 42
    TEST = True

    if TEST:
        FAST = True
        GPU = False
    else:
        FAST = False
        GPU = True

    TEST_ENTRY = (True, True, True)  # (AGG, SEL, COND)

    # 加载用户输入
    test_sql_data, test_table_data, schemas = load_data_for_seq(seq, db_id, FAST=FAST)

    # 加载Word embedding
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), FAST=FAST)

    model = NLNet(word_emb, N_word=N_word, gpu=GPU)

    # 加载模型参数
    model.sel_pred.load_state_dict(torch.load("saved_models/sel_models.dump", map_location='cpu'))
    model.cond_pred.load_state_dict(torch.load("saved_models/cond_models.dump", map_location='cpu'))
    model.group_pred.load_state_dict(torch.load("saved_models/group_models.dump", map_location='cpu'))
    model.order_pred.load_state_dict(torch.load("saved_models/order_models.dump", map_location='cpu'))

    # 输出到文件
    output = "output.txt"
    sql_result = print_one_result(model, test_sql_data, test_table_data, output, schemas, TEST_ENTRY)
    return sql_result


def predict_one_chart(seq, cols):
    N_word = 300
    B_word = 42
    TEST = True

    if TEST:
        FAST = True
        GPU = False
        BATCH_SIZE = 20
    else:
        FAST = False
        GPU = True
        BATCH_SIZE = 64

    # 加载用户输入
    query_seq = load_data_for_chart(seq, FAST=FAST)

    sel_col_seq = []
    sel_col_agg = []
    sel_col_num = []
    for col in cols:
        if "(" in col:
            col_list = list(col)
            st = col_list.index("(")
            ed = col_list.index(")")

            agg = col[0:st]
            col = col[st + 1:ed]
        else:
            agg = "none"
            col = col

        col_res = col.split()
        if "DISTINCT" in col_res:
            col_res.remove("DISTINCT")
        sel_col_seq.append(col_res)
        agg_idx = AGG_OPS.index(agg.lower().strip())
        sel_col_agg.append(agg_idx)
        sel_col_num.append(len(col_res))

    # 加载Word embedding
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), FAST=FAST)

    model = chartNet(word_emb, N_word=N_word, gpu=GPU)

    # 加载模型参数
    # model.chart_pred.load_state_dict(torch.load("saved_models/sel_models.dump", map_location='cpu'))

    # 输出到文件
    output = "chart_output.txt"
    chart_result = print_one_chart(model, [query_seq], [sel_col_seq], [sel_col_agg], [sel_col_num], output)
    return chart_result[0]

def predict_all_sql(dataset):
    N_word = 300
    B_word = 42
    TEST = True

    if TEST:
        FAST = True
        GPU = False
        BATCH_SIZE = 20
    else:
        FAST = False
        GPU = True
        BATCH_SIZE = 64

    TEST_ENTRY = (True, True, True)  # (AGG, SEL, COND)

    # 加载json文件
    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, schemas,\
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(dataset, FAST=FAST)

    # 加载预训练的word embedding
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), FAST=FAST)

    model = NLNet(word_emb, N_word=N_word, gpu=GPU)

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


def predict_charts(dataset):
    print("TODO")
