# -*- coding: UTF-8 -*-
import torch
import traceback
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from collections import defaultdict
from scripts.word_embedding import WordEmbedding
from scripts.model.nlnet.modules.sel_predict import SelPredictor
from scripts.model.nlnet.modules.cond_predict import CondPredictor
from scripts.model.nlnet.modules.group_predict import GroupPredictor
from scripts.model.nlnet.modules.order_predict import OrderPredictor

# define the ops
AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
WHERE_OPS = ['not', 'between', '=', '>', '<', '!=', '>=', '<=', 'in', 'like', 'is', 'exists']
VALUE = '''"VALUE"'''
ORDER_OPS = ['asc limit 1', 'desc limit 1', 'asc', 'desc']


class NLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=120, N_depth=2, gpu=False):
        super(NLNet, self).__init__()

        self.gpu = gpu

        self.N_word = N_word
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_query_num = 200

        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        self.embed_layer = WordEmbedding(word_emb, N_word, gpu)

        self.sel_pred = SelPredictor(N_word, N_h, N_depth, gpu)
        self.cond_pred = CondPredictor(N_word, N_h, N_depth, gpu)
        self.group_pred = GroupPredictor(N_word, N_h, N_depth, gpu)
        self.order_pred = OrderPredictor(N_word, N_h, N_depth, gpu)

        # 不需要sigmod层
        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    '''
        q 为问句的list
        col 为对应的表名组的list
        pred_entry 为目标结果

    '''

    def forward(self, q, col, col_num, q_type, q_concol_seq, pred_entry, gt_where=None, gt_cond=None, gt_sel=None):
        # 问句数目
        B = len(q)

        # 目标结果
        pred_agg, pred_sel, pred_cond = pred_entry

        # 将输入/col字段名转换为词嵌入
        # q部分为分词后组成的list, 因此is_list设置为True
        x_emb_var, x_len = self.embed_layer.gen_x_batch(q_concol_seq, col, is_list=True, is_q=True)
        temp_emb_var, temp_len = self.embed_layer.gen_x_batch(q, col)
        # x_type_emb_var, x_type_len = self.embed_layer.gen_x_batch(q_type, col, is_list=True, is_q=True)
        col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)

        max_x_len = max(x_len)

        # gt_sel = None
        # gt_cond = None

        sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var, col_len, col_name_len, gt_sel=gt_sel)
        cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_len, col_num, col_name_len,  gt_cond=gt_cond, gt_where=gt_where)
        group_score = self.group_pred(x_emb_var, x_len, col_inp_var, col_len, col_num, col_name_len)
        order_score = self.order_pred(x_emb_var, x_len, col_inp_var, col_len, col_num, col_name_len)

        return (sel_score, cond_score, group_score, order_score)

    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_agg, pred_sel, pred_cond = pred_entry

        sel_score, cond_score, group_score, order_score = score

        # 赋值各个部位
        sel_num_score, sel_col_score, agg_num_score, agg_op_score = sel_score
        cond_num_score, cond_col_score, cond_op_score, cond_str_score = cond_score
        gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = group_score
        ody_num_score, ody_col_score, ody_agg_score, ody_par_score = order_score

        B = len(truth_num)
        loss = 0

        #----------loss for sel_pred -------------#

        # loss for sel agg # and sel agg
        # for one batch
        for b in range(len(truth_num)):
            # 涉及到的第一列
            gt_aggs_num = []

            sel_cols = []
            # 遍历每个sel_col
            for i, col in enumerate(truth_num[b][1]):
                if col not in sel_cols:
                    sel_cols.append(col)
                    gt_aggs_num.append(0)
                idx = sel_cols.index(col)
                if truth_num[b][0][i] != 0:
                    gt_aggs_num[idx] += 1

            # print gt_aggs_num
            data = torch.from_numpy(np.array(gt_aggs_num)) #supposed to be gt # of aggs
            if self.gpu:
                agg_num_truth_var = Variable(data.cuda())
            else:
                agg_num_truth_var = Variable(data)
            # num of the select columns
            agg_num_pred = agg_num_score[b, :truth_num[b][5]] # supposed to be gt # of select columns

            if agg_num_pred.size()[0] != agg_num_truth_var.size()[0]:
                print(truth_num[b])

            temp_loss = self.CE(agg_num_pred, agg_num_truth_var)
            loss += (temp_loss / len(truth_num))

            # loss for sel agg prediction
            T = 6 #num agg ops
            # ( B , T ) -> ( unique 的被选中的col数目, 可能的op情况 )
            # 类似onehot的一个列表
            truth_prob = np.zeros((truth_num[b][5], T), dtype=np.float32)
            # [ [被选中行的agg op情况,即curr_sel_aggs] [] ]
            gt_agg_by_sel = []

            sel_cols = []
            for i, col in enumerate(truth_num[b][1]):
                if col not in sel_cols:
                    sel_cols.append(col)
                    gt_agg_by_sel.append([])
                idx = sel_cols.index(col)
                curr_agg = truth_num[b][0][i]
                curr_sel_aggs = gt_agg_by_sel[idx]
                if curr_agg not in curr_sel_aggs:
                    curr_sel_aggs.append(curr_agg)

            for i, col in enumerate(gt_agg_by_sel):
                truth_prob[i][gt_agg_by_sel[i]] = 1

            data = torch.from_numpy(truth_prob)
            if self.gpu:
                agg_op_truth_var = Variable(data.cuda())
            else:
                agg_op_truth_var = Variable(data)
            agg_op_prob = self.sigm(agg_op_score[b, :truth_num[b][5]])

            # 计算target和output之间的二进制交叉熵, 以下为loss function 具体公式
            agg_bce_loss = -torch.mean( 3*(agg_op_truth_var * \
                    torch.log(agg_op_prob+1e-10)) + \
                    (1-agg_op_truth_var) * torch.log(1-agg_op_prob+1e-10) )
            loss += agg_bce_loss / len(truth_num)

        #Evaluate the number of select columns
        sel_num_truth = map(lambda x: x[5]-1, truth_num) #might need to be the length of the set of columms
        temp = list(sel_num_truth)
        data = torch.from_numpy(np.array(temp))
        if self.gpu:
            sel_num_truth_var = Variable(data.cuda())
        else:
            sel_num_truth_var = Variable(data)
        loss += self.CE(sel_num_score, sel_num_truth_var)
        # Evaluate the select columns
        T = len(sel_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth_num[b][1]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            sel_col_truth_var = Variable(data.cuda())
        else:
            sel_col_truth_var = Variable(data)
        sel_col_prob = self.sigm(sel_col_score)
        sel_bce_loss = -torch.mean( 3*(sel_col_truth_var * \
                torch.log(sel_col_prob+1e-10)) + \
                (1-sel_col_truth_var) * torch.log(1-sel_col_prob+1e-10) )
        loss += sel_bce_loss

        # ----------------loss for cond_pred--------------------#
        # cond_num_score, cond_col_score, cond_op_score = cond_score

        # Evaluate the number of conditions
        cond_num_truth = map(lambda x: x[2], truth_num)
        data = torch.from_numpy(np.array(list(cond_num_truth)))
        if self.gpu:
            cond_num_truth_var = Variable(data.cuda())
        else:
            cond_num_truth_var = Variable(data)
        loss += self.CE(cond_num_score, cond_num_truth_var)
        #Evaluate the columns of conditions
        T = len(cond_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][3]) > 0:
                truth_prob[b][list(truth_num[b][3])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            cond_col_truth_var = Variable(data.cuda())
        else:
            cond_col_truth_var = Variable(data)

        cond_col_prob = self.sigm(cond_col_score)
        bce_loss = -torch.mean( 3*(cond_col_truth_var * \
                torch.log(cond_col_prob+1e-10)) + \
                (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
        loss += bce_loss
        #Evaluate the operator of conditions
        for b in range(len(truth_num)):
            if len(truth_num[b][4]) == 0:
                continue
            data = torch.from_numpy(np.array(truth_num[b][4]))
            if self.gpu:
                cond_op_truth_var = Variable(data.cuda())
            else:
                cond_op_truth_var = Variable(data)
            cond_op_pred = cond_op_score[b, :len(truth_num[b][4])]
            # print 'cond_op_truth_var', list(cond_op_truth_var.size())
            # print 'cond_op_pred', list(cond_op_pred.size())
            loss += (self.CE(cond_op_pred, cond_op_truth_var) \
                    / len(truth_num))

        # Evaluate the strings of the cond
        for b in range(len(gt_where)):
            for idx in range(len(gt_where[b])):
                cond_str_truth = gt_where[b][idx]
                if len(cond_str_truth) == 1:
                    continue

                arr = cond_str_score.shape[3]
                flag = False

                for i in cond_str_truth:
                    if i >= arr:
                        flag = True

                if flag:
                    continue
                cond_str_truth_t = [i - 1 if i > 0 else 0 for i in cond_str_truth]
                data = torch.from_numpy(np.array(cond_str_truth_t[1:]))
                if self.gpu:
                    cond_str_truth_var = Variable(data.cuda())
                else:
                    cond_str_truth_var = Variable(data)
                str_end = len(cond_str_truth)-1
                # cond_str_pred [str_end, max_q_len]
                cond_str_pred = cond_str_score[b, idx, :str_end]

                loss += (self.CE(cond_str_pred, cond_str_truth_var) \
                         / (len(gt_where) * len(gt_where[b])))
        # -----------loss for group_pred -------------- #
        #gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = group_score
        # Evaluate the number of group by columns
        gby_num_truth = map(lambda x: x[7], truth_num)
        data = torch.from_numpy(np.array(list(gby_num_truth)))
        if self.gpu:
            gby_num_truth_var = Variable(data.cuda())
        else:
            gby_num_truth_var = Variable(data)
        loss += self.CE(gby_num_score, gby_num_truth_var)
        # Evaluate the group by columns
        T = len(gby_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][6]) > 0:
                truth_prob[b][list(truth_num[b][6])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            gby_col_truth_var = Variable(data.cuda())
        else:
            gby_col_truth_var = Variable(data)
        gby_col_prob = self.sigm(gby_score)
        gby_bce_loss = -torch.mean( 3*(gby_col_truth_var * \
                torch.log(gby_col_prob+1e-10)) + \
                (1-gby_col_truth_var) * torch.log(1-gby_col_prob+1e-10) )
        loss += gby_bce_loss
        # Evaluate having
        having_truth = [1 if len(x[13]) == 1 else 0 for x in truth_num]
        data = torch.from_numpy(np.array(having_truth))
        if self.gpu:
            having_truth_var = Variable(data.cuda())
        else:
            having_truth_var = Variable(data)
        loss += self.CE(hv_score, having_truth_var)
        # Evaluate having col
        T = len(hv_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][13]) > 0:
                truth_prob[b][truth_num[b][13]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            hv_col_truth_var = Variable(data.cuda())
        else:
            hv_col_truth_var = Variable(data)
        hv_col_prob = self.sigm(hv_col_score)
        hv_col_bce_loss = -torch.mean( 3*(hv_col_truth_var * \
                torch.log(hv_col_prob+1e-10)) + \
                (1-hv_col_truth_var) * torch.log(1-hv_col_prob+1e-10) )
        loss += hv_col_bce_loss
        # Evaluate having agg
        T = len(hv_agg_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][12]) > 0:
                truth_prob[b][truth_num[b][12]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            hv_agg_truth_var = Variable(data.cuda())
        else:
            hv_agg_truth_var = Variable(data)
        hv_agg_prob = self.sigm(hv_agg_truth_var)
        hv_agg_bce_loss = -torch.mean( 3*(hv_agg_truth_var * \
                torch.log(hv_agg_prob+1e-10)) + \
                (1-hv_agg_truth_var) * torch.log(1-hv_agg_prob+1e-10) )
        loss += hv_agg_bce_loss
        # Evaluate having op
        T = len(hv_op_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][14]) > 0:
                truth_prob[b][truth_num[b][14]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            hv_op_truth_var = Variable(data.cuda())
        else:
            hv_op_truth_var = Variable(data)
        hv_op_prob = self.sigm(hv_op_truth_var)
        hv_op_bce_loss = -torch.mean( 3*(hv_op_truth_var * \
                torch.log(hv_op_prob+1e-10)) + \
                (1-hv_op_truth_var) * torch.log(1-hv_op_prob+1e-10) )
        loss += hv_op_bce_loss

        # -----------loss for order_pred -------------- #
        #ody_col_score, ody_agg_score, ody_par_score = order_score

        # Evaluate the number of order by columns
        ody_num_truth = map(lambda x: x[10], truth_num)
        data = torch.from_numpy(np.array(list(ody_num_truth)))
        if self.gpu:
            ody_num_truth_var = Variable(data.cuda())
        else:
            ody_num_truth_var = Variable(data)
        loss += self.CE(ody_num_score, ody_num_truth_var)
        # Evaluate the order by columns
        T = len(ody_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][9]) > 0:
                truth_prob[b][list(truth_num[b][9])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            ody_col_truth_var = Variable(data.cuda())
        else:
            ody_col_truth_var = Variable(data)
        ody_col_prob = self.sigm(ody_col_score)
        ody_bce_loss = -torch.mean( 3*(ody_col_truth_var * \
                torch.log(ody_col_prob+1e-10)) + \
                (1-ody_col_truth_var) * torch.log(1-ody_col_prob+1e-10) )
        loss += ody_bce_loss
        # Evaluate order agg assume only one
        T = 6
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][9]) > 0:
                truth_prob[b][list(truth_num[b][8])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            ody_agg_truth_var = Variable(data.cuda())
        else:
            ody_agg_truth_var = Variable(data)
        ody_agg_prob = self.sigm(ody_agg_score)
        ody_agg_bce_loss = -torch.mean( 3*(ody_agg_truth_var * \
                torch.log(ody_agg_prob+1e-10)) + \
                (1-ody_agg_truth_var) * torch.log(1-ody_agg_prob+1e-10) )
        loss += ody_agg_bce_loss
        # Evaluate parity
        ody_par_truth = map(lambda x: x[11], truth_num)
        data = torch.from_numpy(np.array(list(ody_par_truth)))
        if self.gpu:
            ody_par_truth_var = Variable(data.cuda())
        else:
            ody_par_truth_var = Variable(data)
        loss += self.CE(ody_par_score, ody_par_truth_var)
        return loss

    def find_shortest_path(self, start, end, graph):
        stack = [[start, []]]
        visited = set()
        while len(stack) > 0:
            ele, history = stack.pop()
            if ele == end:
                return history
            for node in graph[ele]:
                if node[0] not in visited:
                    stack.append((node[0], history + [(node[0], node[1])]))
                    visited.add(node[0])

    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-': '(',
                       '-RRB-': ')',
                       '-LSB-': '[',
                       '-RSB-': ']',
                       '``': '"',
                       '\'\'': '"',
                       '--': u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        sel_score, cond_score, group_score, order_score = score

        sel_num_score, sel_col_score, agg_num_score, agg_op_score = [x.data.cpu().numpy() if x is not None else None for
                                                                     x in sel_score]
        cond_num_score, cond_col_score, cond_op_score, cond_str_score = [x.data.cpu().numpy() if x is not None else None for x in
                                                         cond_score]
        gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = [
            x.data.cpu().numpy() if x is not None else None for x in group_score]
        ody_num_score, ody_col_score, ody_agg_score, ody_par_score = [x.data.cpu().numpy() if x is not None else None
                                                                      for x in order_score]
        ret_queries = []
        B = len(sel_num_score)
        for b in range(B):
            cur_query = {}

            # Select
            sel_num = np.argmax(sel_num_score[b]) + 1
            cur_query['sel_num'] = sel_num
            cur_query['sel'] = np.argsort(-sel_col_score[b])[:sel_num]

            agg_nums = []
            agg_preds = []
            for idx in range(sel_num):
                curr_num_aggs = np.argmax(agg_num_score[b][idx])
                agg_nums.append(curr_num_aggs)
                if curr_num_aggs == 0:
                    curr_agg_ops = [0]
                else:
                    curr_agg_ops = [x for x in list(np.argsort(-agg_op_score[b][idx])) if x != 0][:curr_num_aggs]
                agg_preds += curr_agg_ops
            cur_query['agg_num'] = agg_nums
            cur_query['agg'] = agg_preds

            # Group by
            gby_num = np.argmax(gby_num_score[b])
            cur_query['gby_num'] = gby_num
            # 'group by' col
            cur_query['group'] = np.argsort(-gby_score[b])[:gby_num]
            cur_query['hv_num'] = np.argmax(hv_score[b])

            if gby_num != 0 and cur_query['hv_num'] != 0:
                cur_query['hv_agg'] = np.argmax(hv_agg_score[b])
                cur_query['hv_col'] = np.argmax(hv_col_score[b])
                cur_query['hv_op'] = np.argmax(hv_op_score[b])
            else:
                cur_query['hv_num'] = 0
                cur_query['hv_agg'] = 0
                cur_query['hv_col'] = -1
                cur_query['hv_op'] = -1

            # Order by
            ody_num = np.argmax(ody_num_score[b])
            cur_query['ody_num'] = ody_num
            cur_query['order'] = np.argsort(-ody_col_score[b])[:ody_num]
            if ody_num != 0:
                cur_query['ody_agg'] = np.argmax(ody_agg_score[b])
                cur_query['parity'] = np.argmax(ody_par_score[b])
            else:
                cur_query['ody_agg'] = 0
                cur_query['parity'] = -1

            # Cond
            cur_query['conds'] = []
            cond_num = np.argmax(cond_num_score[b])
            cond_cols = np.argsort(-cond_col_score[b])[:cond_num]
            all_toks = ['<BEG>'] + q[b] + ['<END>']
            for idx in range(cond_num):
                cur_cond = []
                cur_cond.append(cond_cols[idx])
                cur_cond.append(np.argmax(cond_op_score[b][idx]))
                cur_cond_str_toks = []
                for str_score in cond_str_score[b][idx]:
                    str_tok = np.argmax(str_score[:len(all_toks)])
                    str_val = all_toks[str_tok]
                    if str_val == '<END>':
                        break
                    cur_cond_str_toks.append(str_val)
                cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))

                cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

        return ret_queries

    # (涉及到的表字段, 涉及到的数据库表)
    def gen_from(self, candidate_tables, schema):
        if len(candidate_tables) <= 1:
            org_tables = schema["table_names_original"]
            if len(candidate_tables) == 1:
                temp = list(candidate_tables)
                ret = "from {}".format(org_tables[temp[0]])
            else:
                ret = "from {}".format(org_tables[0])
            # TODO: temporarily settings for select count(*)
            return {}, ret
        # print("candidate:{}".format(candidate_tables))
        table_alias_dict = {}
        uf_dict = {}
        for t in candidate_tables:
            uf_dict[t] = -1
        idx = 1
        graph = defaultdict(list)
        for acol, bcol in schema["foreign_keys"]:
            t1 = schema["column_names"][acol][0]
            t2 = schema["column_names"][bcol][0]
            graph[t1].append((t2, (acol, bcol)))
            graph[t2].append((t1, (bcol, acol)))
        candidate_tables = list(candidate_tables)
        start = candidate_tables[0]
        table_alias_dict[start] = idx
        idx += 1
        ret = "from {} as T1".format(schema["table_names_original"][start])
        try:
            for end in candidate_tables[1:]:
                if end in table_alias_dict:
                    continue
                path = self.find_shortest_path(start, end, graph)
                prev_table = start
                if not path:
                    table_alias_dict[end] = idx
                    idx += 1
                    ret = "{} join {} as T{}".format(ret, schema["table_names_original"][end],
                                                     table_alias_dict[end],
                                                     )
                    continue
                for node, (acol, bcol) in path:
                    if node in table_alias_dict:
                        prev_table = node
                        continue
                    table_alias_dict[node] = idx
                    idx += 1
                    ret = "{} join {} as T{} on T{}.{} = T{}.{}".format(ret, schema["table_names_original"][node],
                                                                        table_alias_dict[node],
                                                                        table_alias_dict[prev_table],
                                                                        schema["column_names_original"][acol][1],
                                                                        table_alias_dict[node],
                                                                        schema["column_names_original"][bcol][1])
                    prev_table = node

        except:

            traceback.print_exc()
            print("db:{}".format(schema["db_id"]))
            # print(table["db_id"])
            return table_alias_dict, ret
        return table_alias_dict, ret

    def gen_sql(self, score, col_org, schema_seq, raw_q, q):

        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-': '(',
                       '-RRB-': ')',
                       '-LSB-': '[',
                       '-RSB-': ']',
                       '``': '"',
                       '\'\'': '"',
                       '--': u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        # 获得各个score情况
        sel_score, cond_score, group_score, order_score = score

        # sel_num (B, 5)
        # sel_col (B, max_col_len)
        # agg_num (B, 5, 5) => (B, sel_num, 每个sel_num对应的agg_num)
        # agg_op (B, 5, 6) => (B, sel_num, 每个sel_num对应的agg_op)
        # todo:中间部分与sel部分相对应?
        sel_num_score, sel_col_score, agg_num_score, agg_op_score = [x.data.cpu().numpy() if x is not None else None for
                                                                     x in sel_score]
        cond_num_score, cond_col_score, cond_op_score, cond_str_score = [x.data.cpu().numpy() if x is not None else None for x in
                                                         cond_score]
        gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = [
            x.data.cpu().numpy() if x is not None else None for x in group_score]
        ody_num_score, ody_col_score, ody_agg_score, ody_par_score = [x.data.cpu().numpy() if x is not None else None
                                                                      for x in order_score]

        ret_queries = []
        ret_sqls = []
        B = len(sel_num_score)
        if B == 1:
            sel_col_score = [sel_col_score]
            agg_num_score = [agg_num_score]
            agg_op_score = [agg_op_score]
            cond_col_score = [cond_col_score]
            cond_op_score = [cond_op_score]
            cond_str_score = [cond_str_score]
            gby_score = [gby_score]
            hv_score = [hv_score]
            hv_col_score = [hv_col_score]
            hv_agg_score = [hv_agg_score]
            hv_op_score = [hv_op_score]
            ody_col_score = [ody_col_score]
            ody_agg_score = [ody_agg_score]
            ody_par_score = [ody_par_score]

        for b in range(B):
            # 对应的字段名
            cur_cols = col_org[b]
            # 对应的schema
            schema = schema_seq[b]

            cur_query = {}

            # for generate sql
            cur_sql = []
            cur_sel = []
            cur_conds = []
            cur_group = []
            cur_order = []
            cur_tables = defaultdict(list)

            # ------------get sel predict
            sel_num_cols = np.argmax(sel_num_score[b]) + 1
            cur_query['sel_num'] = sel_num_cols

            # 取负的原因 -> 获得从大到小的前n个数
            cur_query['sel'] = np.argsort(-sel_col_score[b])[:sel_num_cols]

            agg_nums = []
            agg_preds = []
            agg_preds_gen = []
            for idx in range(sel_num_cols):
                # 可能存在有复数个agg的情况
                curr_num_aggs = np.argmax(agg_num_score[b][idx])
                agg_nums.append(curr_num_aggs)
                if curr_num_aggs == 0:
                    curr_agg_ops = [0]
                else:
                    curr_agg_ops = [x for x in list(np.argsort(-agg_op_score[b][idx])) if x != 0][:curr_num_aggs]
                agg_preds += curr_agg_ops
                agg_preds_gen.append(curr_agg_ops)
            cur_query['agg_num'] = agg_nums
            cur_query['agg'] = agg_preds
            # for gen sel

            cur_sel.append("select")
            for i, cid in enumerate(cur_query['sel']):
                aggs = agg_preds_gen[i]
                agg_num = len(aggs)
                for j, gix in enumerate(aggs):
                    if gix == 0:
                        cur_sel.append([cid, cur_cols[cid][1]])
                        cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                    else:
                        cur_sel.append(AGG_OPS[gix])
                        cur_sel.append("(")
                        cur_sel.append([cid, cur_cols[cid][1]])
                        cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                        cur_sel.append(")")
                    if j < agg_num - 1:
                        cur_sel.append(",")

                if i < sel_num_cols - 1:
                    cur_sel.append(",")

            # ----------get group by predict
            gby_num_cols = np.argmax(gby_num_score[b])
            cur_query['gby_num'] = gby_num_cols
            cur_query['group'] = np.argsort(-gby_score[b])[:gby_num_cols]
            cur_query['hv'] = np.argmax(hv_score[b])
            if gby_num_cols != 0 and cur_query['hv'] != 0:
                cur_query['hv_agg'] = np.argmax(hv_agg_score[b])
                cur_query['hv_col'] = np.argmax(hv_col_score[b])
                cur_query['hv_op'] = np.argmax(hv_op_score[b])
            else:
                cur_query['hv'] = 0
                cur_query['hv_agg'] = 0
                cur_query['hv_col'] = -1
                cur_query['hv_op'] = -1

            # for gen group
            if gby_num_cols > 0:
                cur_group.append("group by")
                for i, gid in enumerate(cur_query['group']):
                    cur_group.append([gid, cur_cols[gid][1]])
                    cur_tables[cur_cols[gid][0]].append([gid, cur_cols[gid][1]])
                    if i < gby_num_cols - 1:
                        cur_group.append(",")
                if cur_query['hv'] != 0:
                    cur_group.append("having")
                    if cur_query['hv_agg'] != 0:
                        cur_group.append(AGG_OPS[cur_query['hv_agg']])
                        cur_group.append("(")
                        cur_group.append([cur_query['hv_col'], cur_cols[cur_query['hv_col']][1]])
                        cur_group.append(")")
                    else:
                        cur_group.append([cur_query['hv_col'], cur_cols[cur_query['hv_col']][1]])
                    cur_tables[cur_cols[cur_query['hv_col']][0]].append(
                        [cur_query['hv_col'], cur_cols[cur_query['hv_col']][1]])
                    cur_group.append(WHERE_OPS[cur_query['hv_op']])
                    cur_group.append(VALUE)

            # --------get order by
            ody_num_cols = np.argmax(ody_num_score[b])
            cur_query['ody_num'] = ody_num_cols
            cur_query['order'] = np.argsort(-ody_col_score[b])[:ody_num_cols]

            # parity指的是 asc,desc 部分
            if ody_num_cols != 0:
                cur_query['ody_agg'] = np.argmax(ody_agg_score[b])
                cur_query['parity'] = np.argmax(ody_par_score[b])
            else:
                cur_query['ody_agg'] = 0
                cur_query['parity'] = -1

            # for gen order
            if ody_num_cols > 0:
                cur_order.append("order by")
                for oid in cur_query['order']:
                    if cur_query['ody_agg'] != 0:
                        cur_order.append(AGG_OPS[cur_query['ody_agg']])
                        cur_order.append("(")
                        cur_order.append([oid, cur_cols[oid][1]])
                        cur_order.append(")")
                    else:
                        cur_order.append([oid, cur_cols[oid][1]])
                    cur_tables[cur_cols[oid][0]].append([oid, cur_cols[oid][1]])

                datid = cur_query['parity']
                if datid == 0:
                    cur_order.append(ORDER_OPS[0])
                elif datid == 1:
                    cur_order.append(ORDER_OPS[1])
                elif datid == 2:
                    cur_order.append(ORDER_OPS[2])
                elif datid == 3:
                    cur_order.append(ORDER_OPS[3])

            # ---------get cond predict
            # cond_num_score, cond_col_score, cond_op_score = [x.data.cpu().numpy() if x is not None else None for x in cond_score]
            cur_query['conds'] = []
            cond_num = np.argmax(cond_num_score[b])
            all_toks = ['<BEG>'] + q[b] + ['<END>']
            max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
            for idx in range(cond_num):

                cur_cond = []
                cur_cond.append(max_idxes[idx])
                cur_cond.append(np.argmax(cond_op_score[b][idx]))
                cur_cond_str_toks = []
                for str_score in cond_str_score[b][idx]:
                    str_tok = np.argmax(str_score[:len(all_toks)])
                    str_val = all_toks[str_tok]
                    if str_val == '<END>':
                        break
                    cur_cond_str_toks.append(str_val)
                cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))

                cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

            # for gen conds
            if len(cur_query['conds']) > 0:
                cur_conds.append("where")
                for i, cond in enumerate(cur_query['conds']):
                    cid, oid, sid = cond
                    cur_conds.append([cid, cur_cols[cid][1]])
                    cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                    cur_conds.append(WHERE_OPS[oid])
                    # todo:VALUE部分!
                    cur_conds.append(sid)
                    if i < cond_num - 1:
                        cur_conds.append("and")

            if -1 in cur_tables.keys():
                del cur_tables[-1]

            table_alias_dict, ret = self.gen_from(cur_tables.keys(), schema)
            # 多表情况
            if len(table_alias_dict) > 0:
                col_map = {}
                for tid, aid in table_alias_dict.items():
                    for cid, col in cur_tables[tid]:
                        col_map[cid] = "t" + str(aid) + "." + col

                new_sel = []
                for s in cur_sel:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_sel.append("*")
                        elif s[0] in col_map:
                            new_sel.append(col_map[s[0]])
                    else:
                        new_sel.append(s)

                new_conds = []
                for s in cur_conds:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_conds.append("*")
                        else:
                            new_conds.append(col_map[s[0]])
                    else:
                        new_conds.append(s)

                new_group = []
                for s in cur_group:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_group.append("*")
                        else:
                            new_group.append(col_map[s[0]])
                    else:
                        new_group.append(s)

                new_order = []
                for s in cur_order:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_order.append("*")
                        else:
                            new_order.append(col_map[s[0]])
                    else:
                        new_order.append(s)

                        # for gen all sql
                cur_sql = new_sel + [ret] + new_conds + new_group + new_order
            else:
                cur_sql = []
                # try:
                # list.extend : 将一个列表中的所有元素加到另一个列表
                cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_sel])
                if len(list(cur_tables.keys())) == 0:
                    cur_tables[0] = []
                cur_sql.extend(["from", schema["table_names_original"][list(cur_tables.keys())[0]]])
                if len(cur_conds) > 0:
                    cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_conds])
                if len(cur_group) > 0:
                    cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_group])
                if len(cur_order) > 0:
                    cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_order])

            sql_str = " ".join(cur_sql)
            ret_sqls.append(sql_str)

        return ret_sqls

    # vis_info : question,cols,sql
    # pred_queries : predict query
    # gt_queries : 已有的query情况
    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry, error_print=False):
        def pretty_print(vis_data, pred_query, gt_query):
            print("\n----------detailed error prints-----------")
            try:
                print('question: %s' % vis_data[0])
                print('question_tok: %s' % vis_data[3])
                print('headers: (%s)'%(' || '.join(vis_data[1])))
                print('query: %s' % vis_data[2])
                print("target query: %s" % gt_query)
                print("pred query: %s"% pred_query)
            except:
                print("\n------skipping print: decoding problem ----------------------")

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                                self.COND_OPS[cond[1]] + ' ' + str(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = 0.0
        sel_err = agg_num_err = agg_op_err = sel_num_err = sel_col_err = 0.0
        cond_err = cond_num_err = cond_col_err = cond_op_err = cond_val_err = 0.0
        gby_err = gby_num_err = gby_col_err = hv_err = hv_col_err = hv_agg_err = hv_op_err = 0.0
        ody_err = ody_num_err = ody_col_err = ody_agg_err = ody_par_err = 0.0

        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

        for b, (pred_qry, gt_qry, vis_data) in enumerate(zip(pred_queries, gt_queries, vis_info)):

            good = True
            tot_flag = True
            sel_flag = True
            cond_flag = True
            gby_flag = True
            ody_flag = True

            # select部分
            sel_gt = gt_qry['sel']
            sel_num_gt = len(set(sel_gt))

            sel_pred = pred_qry['sel']
            sel_num_pred = pred_qry['sel_num']

            if sel_num_pred != sel_num_gt:
                sel_num_err += 1
                sel_flag = False

            # todo: order
            if sorted(set(sel_pred)) != sorted(set(sel_gt)):
                sel_col_err += 1
                sel_flag = False

            agg_gt = gt_qry['agg']
            gt_aggs_num = []
            sel_cols = []
            for i, col in enumerate(gt_qry['sel']):
                if col not in sel_cols:
                    sel_cols.append(col)
                    gt_aggs_num.append(0)
                idx = sel_cols.index(col)
                if agg_gt[i] != 0:
                    gt_aggs_num[idx] += 1

            if gt_aggs_num != pred_qry['agg_num']:
                agg_num_err += 1
                sel_flag = False

            if sorted(gt_qry['agg']) != sorted(pred_qry['agg']):
                agg_op_err += 1
                sel_flag = False

            if not sel_flag:
                sel_err += 1
                good = False

            # group 部分
            gby_gt = gt_qry['group'][:-1]
            gby_pred = pred_qry['group']
            gby_num_pred = pred_qry['gby_num']
            gby_num_gt = len(gby_gt)
            if gby_num_pred != gby_num_gt:
                gby_num_err += 1
                gby_flag = False
            if sorted(gby_pred) != sorted(gby_gt):
                gby_col_err += 1
                gby_flag = False
            gt_gby_agg = gt_qry['group'][-1][0]
            gt_gby_col = gt_qry['group'][-1][1]
            gt_gby_op = gt_qry['group'][-1][2]
            if gby_num_pred != 0 and len(gt_gby_col) != 0:
                if pred_qry['hv_num'] != 1:
                    hv_err += 1
                    gby_flag = False
                if pred_qry['hv_agg'] != gt_gby_agg[0]:
                    hv_agg_err += 1
                    gby_flag = False
                if pred_qry['hv_col'] != gt_gby_col[0]:
                    hv_col_err += 1
                    gby_flag = False
                if pred_qry['hv_op'] != gt_gby_op[0]:
                    hv_op_err += 1
                    gby_flag = False

            if not gby_flag:
                gby_err += 1
                good = False

            # order
            ody_gt_aggs = gt_qry['order'][0]
            ody_gt_cols = gt_qry['order'][1]
            ody_gt_par = gt_qry['order'][2]
            ody_num_cols_pred = pred_qry['ody_num']
            ody_cols_pred = pred_qry['order']
            ody_aggs_pred = pred_qry['ody_agg']
            ody_par_pred = pred_qry['parity']

            if ody_num_cols_pred != len(ody_gt_cols):
                ody_num_err += 1
                ody_flag = False
            if len(ody_gt_cols) > 0:
                if ody_cols_pred != ody_gt_cols:
                    ody_col_err += 1
                    ody_flag = False
                if ody_aggs_pred != ody_gt_aggs:
                    ody_agg_err += 1
                    ody_flag = False
                if ody_par_pred != ody_gt_par:
                    ody_par_err += 1
                    ody_flag = False

            if not ody_flag:
                ody_err += 1
                good = False

            # conds
            cond_pred = pred_qry['conds']
            cond_gt = gt_qry['where']
            flag = True
            if len(cond_pred) != len(cond_gt):
                flag = False
                cond_num_err += 1
                cond_flag = False
            if flag and set(x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                flag = False
                cond_col_err += 1
                cond_flag = False
            for idx in range(len(cond_pred)):
                if not flag:
                    break
                gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                    flag = False
                    cond_op_err += 1
                    cond_flag = False

            for idx in range(len(cond_pred)):
                if not flag:
                    break
                gt_idx = tuple(
                    x[0] for x in cond_gt).index(cond_pred[idx][0])
                if flag and str(cond_gt[gt_idx][2]).lower() != \
                        str(cond_pred[idx][2]).lower():
                    flag = False
                    cond_val_err += 1
                    cond_flag = False

            if not cond_flag:
                cond_err += 1
                good = False

            if not good:
                # 输出错误的query
                if error_print:
                    pretty_print(vis_data, pred_qry, gt_qry)
                tot_err += 1

        return np.array((sel_err, cond_err, gby_err, ody_err)), tot_err


    def generate_gt_where_seq(self, q, col, query):
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            cur_values = []
            st = cur_query.index(u'WHERE') + 1 if \
                u'WHERE' in cur_query else len(cur_query)
            all_toks = ['<BEG>'] + cur_q + ['<END>']
            while st < len(cur_query):
                if 'INTERSECT' in cur_query[st:] or 'EXCEPT' in cur_query[st:] or 'UNION' in cur_query[st:]:
                    break

                if 'AND' in cur_query[st:] or 'OR' in cur_query[st:]:
                    if 'AND' in cur_query[st:]:
                        ed = cur_query[st:].index('AND') + st
                        if cur_query[st:].index('AND') > 8:
                            ed = ed - 5
                    else:
                        ed = cur_query[st:].index('OR') + st
                        if cur_query[st:].index('OR') > 8:
                            ed = ed - 5
                else:
                    # ed = len(cur_query)
                    ed = st + 5
                if '=' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('=') + st
                elif '>' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('>') + st
                elif '<' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('<') + st
                else:
                    st = ed + 1
                    continue
                    # raise RuntimeError("No operator in it!")
                temp = cur_query[op + 1:ed]
                if "'" in temp and temp.index("'") != 0:
                    idx = temp.index("'")
                    idx0 = idx - 1
                    temp[idx0] = temp[idx0] + temp[idx]
                    temp.remove("'")
                this_str = ['<BEG>'] + temp + ['<END>']
                cur_seq = [all_toks.index(s) if s in all_toks \
                               else 0 for s in this_str]
                cur_values.append(cur_seq)
                st = ed + 1
            ret_seq.append(cur_values)
        return ret_seq

