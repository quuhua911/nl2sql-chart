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

def load_data_for_chart(seq, FAST=True):
    tokens = nltk.word_tokenize(seq)
    return tokens


def load_data_for_seq(seq, db_id, FAST=True):
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


def load_word_emb(file, load_used=False, FAST=True):
    if not load_used:
        ret = {}

        with open(file) as file_inf:
            # 获得每一个字符的embedding情况
            for idx, line in enumerate(file_inf):
                if FAST and idx >= 2000:
                    break
                info = line.strip().split(' ')
                # 首位代表字符
                if info[0].lower() not in ret:
                    line_d = map(lambda x: float(x), info[1:])
                    temp = list(line_d)
                    ret[info[0]] = np.array(temp)
            return ret
    else:
        with open('../glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('../glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val

# 加载数据集
def load_dataset(dataset, FAST=True):
    print("Loading from datasets...")
    dir = dataset
    # train.json
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

        if "type_of_chart" in query:
            temp["type_of_chart"] = query["type_of_chart"]
            temp["x_col"] = query["x_col"]
            temp["y_col"] = query["y_col"]
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
        gt_sel = table_sel[1]
        if len(gt_sel) > 3:
            gt_sel = gt_sel[:3]
        for tup in gt_sel:
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

        # process intersect/except/union
        temp['special'] = 0
        if sql['intersect'] is not None:
            temp['special'] = 1
        elif sql['except'] is not None:
            temp['special'] = 2
        elif sql['union'] is not None:
            temp['special'] = 3

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

        # query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)

        sel_col_seq = []
        sel_col_agg = []
        sel_col_num = []
        for idx in range(len(col_seq)):
            curr_sel = ans_seq[idx][1]
            curr_table = col_seq[idx]
            curr_sel_col = [curr_table[x] for x in curr_sel]
            curr_sel_agg = [ans_seq[idx][0][i] for i, x in enumerate(curr_sel)]
            sel_col_seq.append(curr_sel_col)
            sel_col_agg.append(curr_sel_agg)
            sel_col_num.append(len(curr_sel_col))

        if type(model).__name__ == "chartNet":
            score = model.forward(query_seq, sel_col_seq, sel_col_agg, sel_col_num)
            charts = model.gen_chart_info(score, query_seq, sel_col_seq, sel_col_agg)
            for chart_info in charts:
                output.write(str(chart_info["type"]) + " "
                             + str(chart_info["x_col"]) + " "
                             + str(chart_info["y_col"]) + "\n")
        else:
            score = model.forward(q_seq, col_seq, col_num, pred_entry)
            gen_sqls = model.gen_sql(score, col_org_seq, schema_seq)
            for sql in gen_sqls:
                output.write(sql+"\n")
        st = ed

def print_one_chart(model, query_seq, sel_col_seq, sel_col_agg, sel_col_num, output_file):
    model.eval()
    output = open(output_file, 'w')
    score = model.forward(query_seq, sel_col_seq, sel_col_agg, sel_col_num)
    chart = model.gen_chart_info(score, query_seq, sel_col_seq, sel_col_agg)

    for chart_info in chart:
        output.write(str(chart_info["type"]) + " "
                     + str(chart_info["x_col"]) + " "
                     + str(chart_info["y_col"]) + "\n")
    return chart


def print_one_result(model, sql_data, table_data, output_file, schemas, pred_entry):
    model.eval()
    output = open(output_file, 'w')

    q_seq, col_seq, col_num, col_org_seq, schema_seq = to_one_seq(sql_data, table_data, schemas)
    score = model.forward(q_seq, col_seq, col_num, pred_entry)
    gen_sqls = model.gen_sql(score, col_org_seq, schema_seq)
    for sql in gen_sqls:
        output.write(sql+"\n")
    return gen_sqls


def to_one_seq(sql, table_data, schemas):
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
    for i in range(st, ed):
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

        query_gt, _ = to_batch_query(sql_data, perm, st, ed)

        sel_col_seq = []
        sel_col_agg = []
        sel_col_num = []
        for idx in range(len(col_seq)):
            curr_sel = ans_seq[idx][1]
            curr_table = col_seq[idx]
            curr_sel_col = [curr_table[x] for x in curr_sel]
            curr_sel_agg = [ans_seq[idx][0][i] for i, x in enumerate(curr_sel)]
            sel_col_seq.append(curr_sel_col)
            sel_col_agg.append(curr_sel_agg)
            sel_col_num.append(len(curr_sel_col))

        if type(model).__name__ == 'chartNet':
            temp = query_gt[:]
            rev_num = 0
            for idx in range(len(query_gt)):
                if int(temp[idx]['type_of_chart']) > 2:
                    new_idx = idx - rev_num
                    query_seq.pop(new_idx)
                    query_gt.pop(new_idx)
                    sel_col_seq.pop(new_idx)
                    sel_col_agg.pop(new_idx)
                    sel_col_num.pop(new_idx)
                    rev_num += 1
            score = model.forward(query_seq, sel_col_seq, sel_col_agg, sel_col_num)
            loss = model.loss(score, query_gt)
        else:
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
        ed = st + batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, \
        raw_data, col_org_seq, schema_seq = \
            to_batch_seq(sql_data, table_data, perm, st, ed, schemas, ret_vis_data=True)

        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)

        sel_col_seq = []
        sel_col_agg = []
        sel_col_num = []
        for idx in range(len(col_seq)):
            curr_sel = ans_seq[idx][1]
            curr_table = col_seq[idx]
            curr_sel_col = [curr_table[x] for x in curr_sel]
            curr_sel_agg = [ans_seq[idx][0][i] for i, x in enumerate(curr_sel)]
            sel_col_seq.append(curr_sel_col)
            sel_col_agg.append(curr_sel_agg)
            sel_col_num.append(len(curr_sel_col))

        if type(model).__name__ == "chartNet":
            temp = query_gt[:]
            rev_num = 0
            for idx in range(len(query_gt)):
                if int(temp[idx]['type_of_chart']) > 2:
                    new_idx = idx - rev_num
                    query_seq.pop(new_idx)
                    query_gt.pop(new_idx)
                    sel_col_seq.pop(new_idx)
                    sel_col_agg.pop(new_idx)
                    sel_col_num.pop(new_idx)
                    rev_num += 1
            score = model.forward(query_seq, sel_col_seq, sel_col_agg, sel_col_num)
            pred_list = model.gen_chart_info(score, query_seq, sel_col_seq, sel_col_agg)
            one_err, tot_err = model.check_acc(pred_list, query_gt, error_print)
        else:
            score = model.forward(q_seq, col_seq, col_num, pred_entry)
            pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, pred_entry)
            one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry, error_print)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)
        st = ed

    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

