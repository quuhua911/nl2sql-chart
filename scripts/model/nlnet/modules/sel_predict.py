# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
from scripts.net_utils import run_lstm, col_name_encode

class SelPredictor(nn.Module):
    # N_h 隐藏元维数
    # N_word 输入数据向量维数，指词嵌入维数
    # N_depth LSTM层数，串联的LSTM数目
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(SelPredictor, self).__init__()
        self.gpu = gpu

        self.N_h = N_h

        # Encoding
        # Query word embedding
        # 因为是双向lstm, 需要将hidden_size/2
        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                              num_layers=N_depth, batch_first=True,
                              dropout=0.3, bidirectional=True)

        # col word embedding
        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                              num_layers=N_depth, batch_first=True,
                              dropout=0.3, bidirectional=True)

        # nn.Linear(input_size,output_size)

        # Decoding
        # col 数目部分
        self.q_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(N_h, N_h)
        # num of cols :1-4
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5))

        # 具体col部分
        self.q_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)

        # 最后是输出维数是1 , 因此可通过squeeze消去
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        # agg数目部分
        self.agg_num_att = nn.Linear(N_h, N_h)
        self.agg_num_out_q = nn.Linear(N_h, N_h)
        self.agg_num_out_c = nn.Linear(N_h, N_h)
        self.agg_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5))

        # 具体agg部分
        self.agg_att = nn.Linear(N_h, N_h)
        self.agg_out_q = nn.Linear(N_h, N_h)
        self.agg_out_c = nn.Linear(N_h, N_h)
        # 可能的agg为 [ none, max, min, count, sum, avg ]
        self.agg_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6))

        self.softmax = nn.Softmax(dim=1)
        # 各损失函数

        # 适用于softmax函数
        self.CE = nn.CrossEntropyLoss()

        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    # todo:gt_sel?
    def forward(self, q_emb_var, q_len, col_emb_var, col_len, col_name_len, gt_sel):
        max_q_len = max(q_len)
        max_col_len = max(col_len)

        # 输入数目
        B = len(q_len)

        # Encode query
        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)

        # Encode col
        col_enc, _ = col_name_encode(self.col_lstm, col_emb_var, col_name_len, col_len)

        '''
            Predict col num
        '''
        # att_val_qc_num: (B, max_col_len, max_q_len)
        '''
            bmm 实现对于batch的矩阵乘法
            transpose 实现转置
        '''
        temp = self.q_num_att(q_enc).transpose(1, 2)
        att_val_qc_num = torch.bmm(col_enc, temp)

        # 将未满足最大长度的部分做消去处理
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_val_qc_num[idx, num:, :] = -100

        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc_num[idx, :, num:] = -100

        # view()中-1代表自动计算行/列数，一个view函数只能有一个-1参数
        # 三维用(-1,x)表示一二维合并
        # TODO(closed):直接softmax(...(B,-1,max_q_len))和下述方法的区别 : 后者进行了一次重排
        # TODO: 基于 max_q_len softmax的原因?
        # softmax 应用于 q_len 即可
        combine_temp = att_val_qc_num.view((-1, max_q_len))
        qc_num_softmax = self.softmax(combine_temp)
        # 复原
        att_prob_qc_num = qc_num_softmax.view(B, -1, max_q_len)

        # q_weighted_num: (B, hid_dim)
        # squeeze(num)/unsqueeze(num) 消除idx为num的维数,该维数必须为1否则函数没有效果
        mul_enc_att = (q_enc.unsqueeze(1) * att_prob_qc_num.unsqueeze(3))

        temp_sum = mul_enc_att.sum(2)
        q_weighted_num = temp_sum.sum(1)

        # self.col_num_out: (B,4)
        col_num_score = self.col_num_out(self.col_num_out_q(q_weighted_num))

        '''
            Predict columns
        '''
        att_val_qc = torch.bmm(col_enc, self.q_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, :, num:] = -100
        att_prob_qc = self.softmax(att_val_qc.view((-1, max_q_len))).view(B, -1, max_q_len)

        # q_weighted : (B,max_col_len, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qc.unsqueeze(3)).sum(2)

        # Compute prediction scores
        # self.col_out.sequeeze(): (B, max_col_len)
        col_out_temp = self.col_out_q(q_weighted) + self.col_out_c(col_enc)
        col_score = self.col_out(col_out_temp).squeeze()
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                # 不足的长度部分补足
                col_score[idx, num:] = -100

        '''
            get select columns for agg prediction
            The part before only chooses the possible columns and this part will choose the columns which will have agg function
        '''

        # todo: chosen_sel_gt 的 作用
        chosen_sel_gt = []
        if gt_sel is None:
            sel_nums_list = list(np.argmax(col_num_score.data.cpu().numpy(), axis=1))
            sel_nums = [x+1 for x in sel_nums_list]
            sel_col_scores = col_score.data.cpu().numpy()
            chosen_sel_gt = [list(np.argsort(-sel_col_scores[b])[:sel_nums[b]])
                             for b in range(len(sel_nums))]
        else:
            for x in gt_sel:
                curr = x[0]
                curr_sel = [curr]
                for col in x:
                    if col != curr:
                        curr_sel.append(col)
                chosen_sel_gt.append(curr_sel)

        col_emb = []
        for b in range(B):
            temp = [col_enc[b, x] for x in chosen_sel_gt[b]]
            # Join a sequence of arrays along a new axis.
            cur_col_emb = torch.stack(temp + [col_enc[b, 0]] * (5 - len(chosen_sel_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)  # (B, 4, hide)

        '''
            Predict aggregation
        '''

        # q_enc.unsqueeze(1): (B, 1, max_x_len, hd)
        # col_emb.unsqueeze(3): (B, 4, hd, 1)
        # agg_num_att_val.squeeze: (B, 4, max_x_len)

        # predict agg_num
        agg_num_att_val = torch.matmul(self.agg_num_att(q_enc).unsqueeze(1),
                                       col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                agg_num_att_val[idx, :, num:] = -100
        agg_num_att = self.softmax(agg_num_att_val.view(-1, max_q_len)).view(B, -1, max_q_len)
        q_weighted_agg_num = (q_enc.unsqueeze(1) * agg_num_att.unsqueeze(3)).sum(2)
        # (B, 4, 4)
        agg_num_score = self.agg_num_out(self.agg_num_out_q(q_weighted_agg_num) +
                                         self.agg_num_out_c(col_emb)).squeeze()

        # predict agg_col
        agg_att_val = torch.matmul(self.agg_att(q_enc).unsqueeze(1),
                                   col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                agg_att_val[idx, :, num:] = -100
        agg_att = self.softmax(agg_att_val.view(-1, max_q_len)).view(B, -1, max_q_len)
        q_weighted_agg = (q_enc.unsqueeze(1) * agg_att.unsqueeze(3)).sum(2)

        agg_score = self.agg_out(self.agg_out_q(q_weighted_agg) +
                                 self.agg_out_c(col_emb)).squeeze()

        score = (col_num_score, col_score, agg_num_score, agg_score)

        return score
