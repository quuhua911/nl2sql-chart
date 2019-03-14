# -*- coding: UTF-8 -*-

import re
import io
import json
import nltk
import numpy as np
import os

'''
    question:
    question_toks:
    db_id:
    query:
    query_toks:
    sql:
'''


def load_test_seq(seq, db_id, FAST=True):
    # with open(seq) as seq_inf:
    #   seq_json = json.load(seq_inf)
    tokens = nltk.word_tokenize(seq)

    test_seq_json = {
        "db_id": db_id,
        "question": seq,
        "question_toks": tokens
    }

    dir = "data/"
    TABLE_DIR = os.path.join(dir, "tables.json")

    with open(TABLE_DIR) as table_inf:
        print("Loading data from %s" % TABLE_DIR)
        table_origin_data = json.load(table_inf)

    tables = {}
    table_cols = {}

    for i in range(len(table_origin_data)):
        table = table_origin_data[i]
        temp = {}
        temp['cols'] = table['column_names']

        db_name = table['db_id']
        table_cols[db_name] = temp
        tables[db_name] = table

    temp = {}

    query = test_seq_json
    # 单个data基本信息
    temp['question'] = query["question"]
    temp['question_tok'] = query['question_toks']
    # 省去了具体值
    temp['table_id'] = query['db_id']

    table = tables[temp['table_id']]
    temp['col_original'] = table["column_names_original"]
    temp['table_original'] = table["table_names_original"]
    temp['foreign_key'] = table['foreign_keys']

    return temp, table_cols, tables


def load_word_emb(file, FAST=True):
    ret = {}

    with open(file) as file_inf:
        # 获得每一个字符的embedding情况
        for idx, line in enumerate(file_inf):
            if FAST and idx >= 2000:
                break
            info = line.strip().split(' ')
            # 首位代表字符
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(map(lambda x: float(x), info[1:]))

        return ret


# 加载数据集
def load_dataset(dataset, FAST=True):
    print("Loading from datasets...")
    dir = dataset
    TABLE_DIR = os.path.join(dir, "tables.json")
    TRAIN_DIR = os.path.join(dir, "train.json")
    DEV_DIR = os.path.join(dir, "dev.json")
    TEST_DIR = os.path.join(dir, "dev.json")

    with open(TABLE_DIR) as table_inf:
        print("Loading data from %s" % TABLE_DIR)
        table_origin_data = json.load(table_inf)

    train_sql_data, train_table_data, schemas_all = load_dataset_with_format(TRAIN_DIR, table_origin_data, FAST)
    dev_sql_data, dev_table_data, schemas = load_dataset_with_format(DEV_DIR, table_origin_data, FAST)
    test_sql_data, test_table_data, schemas = load_dataset_with_format(TEST_DIR, table_origin_data, FAST)

    TRAIN_DB = 'data/train.db'
    DEV_DB = 'data/dev.db'
    TEST_DB = 'data/test.db'

    # schemas_all = None

    return train_sql_data, train_table_data, dev_sql_data, dev_table_data, \
           test_sql_data, test_table_data, schemas_all, TRAIN_DB, DEV_DB, TEST_DB


# 读取数据集并进行格式化
def load_dataset_with_format(dir, table_origin_data, FAST):

    with open(dir) as dataset_inf:
        print("Loading data from %s" % dir)
        dataset_origin_data = json.load(dataset_inf)

    sql_data, table_data = format_dataset(dataset_origin_data, table_origin_data)

    schemas = {}
    for table in table_origin_data:
        schemas[table["db_id"]] = table

    if FAST:
        return sql_data[:80], table_data, schemas
    else:
        return sql_data, table_data, schemas


# 格式化数据集
def format_dataset(dataset_origin_data, table_origin_data):
    tables = {}
    table_cols = {}
    sqls = []

    for i in range(len(table_origin_data)):
        table = table_origin_data[i]
        temp= {}
        temp['cols'] = table['column_names']

        db_name = table['db_id']
        table_cols[db_name] = temp
        tables[db_name] = table

    for i in range(len(dataset_origin_data)):
        query = dataset_origin_data[i]
        temp={}

        # 单个data基本信息
        temp['question'] = query["question"]
        temp['query'] = query['query']
        temp['question_tok'] = query['question_toks']
        # 省去了具体值
        temp['query_tok'] = query['query_toks']
        temp['table_id'] = query['db_id']

        table = tables[temp['table_id']]
        temp['col_original'] = table["column_names_original"]
        temp['table_original'] = table["table_names_original"]
        temp['foreign_key'] = table['foreign_keys']

        # 处理sql信息
        sql = query['sql']

        # SELECT部分
        temp['sel'] = []
        temp['agg'] = []

        '''
            "select": [
                false, # is distinct
                [
                    [
                        3, # agg op index
                        [
                            0, # saved for col operation
                            [
                                0, # agg op index
                                0, # index of col
                                false # is distinct
                            ],
                            null # second col for col operation
                        ]
                    ]
                ]
            ],
        '''

        table_sel = sql['select']
        for tup in table_sel[1]:
            temp['agg'].append(tup[0])
            temp['sel'].append(tup[1][1][1])

        # WHERE部分
        temp['where'] = []

        '''
            "where": [
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            13,
                            false
                        ],
                        null
                    ],
                    "\"Yes\"",
                    null
                ]
            ]
        '''

        table_where = sql['where']
        if len(table_where) > 0:
            # 奇数位是'and'/'or'
            conds = [table_where[x] for x in range(len(table_where)) if x % 2 == 0]
            for cond in conds:
                temp_cond = []

                temp_cond.append(cond[2][1][1])

                temp_cond.append(cond[1])

                if cond[4] is not None:
                    temp_cond.append([cond[3], cond[4]])
                else:
                    temp_cond.append(cond[3])

                temp['where'].append(temp_cond)

        temp['conj'] = [table_where[x] for x in range(len(table_where)) if x % 2 == 1]

        # GROUP BY部分
        temp['group'] = [x[1] for x in sql['groupBy']]  # assume only one groupby
        having_cond = []
        if len(sql['having']) > 0:
            gt_having = sql['having'][0]  # currently only do first having condition
            having_cond.append([gt_having[2][1][0]])  # aggregator
            having_cond.append([gt_having[2][1][1]])  # column
            having_cond.append([gt_having[1]])  # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        else:
            having_cond = [[], [], []]
        temp['group'].append(having_cond)  # GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

        # ORDER BY部分
        order_aggs = []
        order_cols = []
        temp['order'] = []
        order_par = 4
        gt_order = sql['orderBy']
        limit = sql['limit']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1][:1]]  # limit to 1 order by
            order_cols = [x[1][1] for x in gt_order[1][:1]]
            if limit != None:
                if gt_order[0] == 'asc':
                    order_par = 0
                else:
                    order_par = 1
            else:
                if gt_order[0] == 'asc':
                    order_par = 2
                else:
                    order_par = 3

        temp['order'] = [order_aggs, order_cols, order_par]  # GOLD for ORDER [[[agg], [col], [dat]], []]

        sqls.append(temp)

    return sqls, table_cols


def print_results(model, batch_size, sql_data, table_data, output_file, schemas, pred_entry, error_print=False, train_flag = False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    output = open(output_file, 'w')
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq,\
         raw_data, col_org_seq, schema_seq = to_batch_seq(sql_data, table_data, perm, st, ed, schemas, ret_vis_data=True)
        score = model.forward(q_seq, col_seq, col_num, pred_entry)
        gen_sqls = model.gen_sql(score, col_org_seq, schema_seq)
        for sql in gen_sqls:
            output.write(sql+"\n")
        st = ed


def print_input_result(model, batch_size, sql_data, table_data, output_file, schemas, pred_entry, error_print=False, train_flag = False):
    model.eval()
    output =  open(output_file, 'w')

    q_seq, col_seq, col_num, col_org_seq, schema_seq = to_input_seq(sql_data, table_data, schemas)
    score = model.forward(q_seq, col_seq, col_num, pred_entry)
    gen_sqls = model.gen_sql(score, col_org_seq, schema_seq)
    for sql in gen_sqls:
        output.write(sql+"\n")


def to_input_seq(sql, table_data, schemas):
    q_seq = []
    col_seq = []
    col_num = []
    col_org_seq = []
    schema_seq = []

    col_org_seq.append(sql['col_original'])
    q_seq.append(sql['question_tok'])
    table = table_data[sql['table_id']]
    schema_seq.append(schemas[sql['table_id']])
    col_num.append(len(table['cols']))
    tab_cols = [col[1] for col in table['cols']]
    col_seq.append([x.split(" ") for x in tab_cols])

    return q_seq, col_seq, col_num, col_org_seq, schema_seq


def to_batch_query(sql_data, perm, st, ed):
    query_gt = []
    table_ids = []
    for i in range(len(sql_data)):
        query_gt.append(sql_data[perm[i]])
        table_ids.append(sql_data[perm[i]]['table_id'])
    return query_gt, table_ids


def to_batch_seq(sql_data, table_data, idxes, st, ed, schemas, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []

    col_org_seq = []
    schema_seq = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        col_org_seq.append(sql['col_original'])
        q_seq.append(sql['question_tok'])
        table = table_data[sql['table_id']]
        schema_seq.append(schemas[sql['table_id']])
        col_num.append(len(table['cols']))
        tab_cols = [col[1] for col in table['cols']]
        col_seq.append([x.split(" ") for x in tab_cols])
        ans_seq.append((sql['agg'],     # sel agg # 0
            sql['sel'],                 # sel col # 1
            len(sql['where']),           # cond # 2
            tuple(x[0] for x in sql['where']), # cond col 3
            tuple(x[1] for x in sql['where']), # cond op 4
            len(set(sql['sel'])),       # number of unique select cols 5
            sql['group'][:-1],          # group by columns 6
            len(sql['group']) - 1,      # number of group by columns 7
            sql['order'][0],            # order by aggregations 8
            sql['order'][1],            # order by columns 9
            len(sql['order'][1]),       # num order by columns 10
            sql['order'][2],            # order by parity 11
            sql['group'][-1][0],        # having agg 12
            sql['group'][-1][1],        # having col 13
            sql['group'][-1][2]         # having op 14
            ))
        #order: [[agg], [col], [dat]]
        #group: [col1, col2, [agg, col, op]]
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append([x for x in sql['where']])
        vis_seq.append((sql['question'], tab_cols, sql['query']))

    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq, col_org_seq, schema_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, col_org_seq, schema_seq


def epoch_train(model, optimizer, batch_size, sql_data, table_data, schemas, pred_entry):
    model.train()
    # permutation() 随机打乱数组
    perm = np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, col_org_seq, schema_seq = \
            to_batch_seq(sql_data, table_data, perm, st, ed, schemas)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry)
        cum_loss += loss.data.cpu().numpy().tolist() * (ed - st)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)


def epoch_acc(model, batch_size, sql_data, table_data, schemas, pred_entry, error_print=False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st + batch_size if st+batch_size<len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, \
        raw_data, col_org_seq, schema_seq = \
            to_batch_seq(sql_data, table_data, perm, st, ed, schemas, ret_vis_data=True)

        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        score = model.forward(q_seq, col_seq, col_num, pred_entry)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, pred_entry)

        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry, error_print)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)
        st = ed

    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

