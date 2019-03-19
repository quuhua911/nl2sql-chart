import json
import torch
import datetime
import argparse
import numpy as np
from scripts.utils import *
from scripts.model.chartnet.chartnet import chartNet


if __name__ == '__main__':
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

    dataset_dir = "data/"
    saved_models_dir = "saved_models/"

    # 加载json文件
    sql_data, table_data, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, schemas, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(dataset_dir, FAST=FAST)

    # 加载预训练的word embedding
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), FAST=FAST)

    model = chartNet(word_emb, N_word=N_word, gpu=GPU)

    print("Loading from chart model...")
    model.chart_pred.load_state_dict(torch.load(saved_models_dir + "chart_models.dump", map_location='cpu'))

    output = "chart_output.txt"
    print_results(model, BATCH_SIZE, test_sql_data, test_table_data, output, schemas, TEST_ENTRY)