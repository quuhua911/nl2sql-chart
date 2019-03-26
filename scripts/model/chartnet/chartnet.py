# -*- coding = UTF-8 -*-
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from scripts.word_embedding import WordEmbedding
from scripts.model.nlnet.modules.sel_predict import SelPredictor
from scripts.model.chartnet.modules.chart_predict import ChartPredictor


AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
class chartNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=120, N_depth=2, gpu=False):
        super(chartNet, self).__init__()

        self.gpu = gpu
        self.N_word = N_word
        self.N_h = N_h
        self.N_depth = N_depth

        self.embed_layer = WordEmbedding(word_emb, N_word, gpu)
        self.chart_pred = ChartPredictor(N_word, N_h, N_depth, gpu)

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    def forward(self, q, col, col_agg, col_num):
        B = len(q)

        # todo:encode the agg
        x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
        col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)

        chart_score = self.chart_pred(x_emb_var, x_len, col_inp_var, col_len, col_name_len)

        return chart_score

    def loss(self, score, truth):

        type_score, x_col_score, y_col_score = score

        loss = 0

        # for b in range(len(truth)):
        # Type部分
        type_truth = map(lambda x: int(x['type_of_chart']), truth)
        temp = list(type_truth)
        # 数据集
        temp_truth = [x if x < 3 else 0 for x in temp]
        data = torch.from_numpy(np.array(temp_truth))

        if self.gpu:
            type_truth_var = Variable(data.cuda())
        else:
            type_truth_var = Variable(data)

        ce = self.CE(type_score, type_truth_var)

        loss += ce

        # x_col及y_col部分
        x_truth = []
        y_truth = []
        # x_pred = []
        # y_pred = []

        # 消除type为0时的x/y情况

        for idx, type in enumerate(temp_truth):
            #if type == 0:
            #    continue
            # x_pred.append(x_col_score[idx, :].tolist())
            # y_pred.append(y_col_score[idx, :].tolist())
            x_truth.append(int(truth[idx]['x_col']))
            y_truth.append(int(truth[idx]['y_col']))

        # x_pred = torch.tensor(x_pred, dtype=torch.float32)
        # y_pred = torch.tensor(y_pred, dtype=torch.float32)


        data = torch.from_numpy(np.array(x_truth))
        if self.gpu:
            #x_pred_var = Variable(x_pred.cuda())
            x_truth_var = Variable(data.cuda())
        else:
            #x_pred_var = Variable(x_pred)
            x_truth_var = Variable(data)

        loss = loss + self.CE(x_col_score, x_truth_var)

        data = torch.from_numpy(np.array(y_truth))
        if self.gpu:
            # y_pred_var = Variable(y_pred.cuda())
            y_truth_var = Variable(data.cuda())
        else:
            # y_pred_var = Variable(y_pred)
            y_truth_var = Variable(data)
        loss = loss + self.CE(y_col_score, y_truth_var)

        return loss

    def gen_chart_info(self, score, sql, col, col_agg):
        type_score, x_col_score, y_col_score = score

        type_score_c = type_score.data.cpu().numpy()
        x_col_score_c = x_col_score.data.cpu().numpy()
        y_col_score_c = y_col_score.data.cpu().numpy()

        B = len(type_score)

        ret_lists = []
        for b in range(B):
            cur_list = {}

            type = np.argmax(type_score_c[b])
            cur_list["type"] = type

            x_col = np.argmax(x_col_score_c[b])
            cur_list["x_col"] = x_col

            y_col = list(np.argsort(-y_col_score_c[b])[:2])
            if y_col[0] == x_col:
                cur_list["y_col"] = y_col[1]
            else:
                cur_list["y_col"] = y_col[0]

            ret_lists.append(cur_list)
        return ret_lists

    def check_acc(self, pred_lists, gt_lists, error_print=False):
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

        B = len(gt_lists)

        tot_err = 0.0
        type_err = x_col_err = y_col_err = 0.0

        for b, (pred_list, gt_list) in enumerate(zip(pred_lists, gt_lists)):
            good = True

            # type_flag = True
            # x_flag = True
            # y_flag = True

            type_gt = gt_list['type_of_chart']

            type_pred = pred_list['type']

            if type_gt != type_pred:
                type_err = type_err + 1
                good = False

            x_gt = gt_list['x_col']
            x_pred = pred_list['x_col']

            if x_gt != x_pred:
                x_col_err = x_col_err + 1
                good = False

            y_gt = gt_list['y_col']
            y_pred = pred_list['y_col']

            if y_gt != y_pred:
                y_col_err = y_col_err + 1
                good = False

            if not good:
                tot_err += 1

        return np.array((type_err, x_col_err, y_col_err)), tot_err
