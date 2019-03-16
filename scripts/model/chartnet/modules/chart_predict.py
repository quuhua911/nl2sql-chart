# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
from scripts.net_utils import run_lstm, col_name_encode

class ChartPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(ChartPredictor, self).__init__()
        self.gpu = gpu

        self.N_h = N_h

        # Encoding
        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                              num_layers=N_depth, batch_first=True,
                              dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                                num_layers=N_depth, batch_first=True,
                                dropout=0.3, bidirectional=True)

        # Decoding

        # type类型部分
        self.type_att = nn.Linear(N_h, N_h)
        self.type_out_q = nn.Linear(N_h, N_h)
        self.type_out_c = nn.Linear(N_h, N_h)
        # possible classes: [ 0, 1, 2 ] => [ 'None', 'Bar', 'Line' ]
        self.type_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 3))

        # X col部分
        self.x_att = nn.Linear(N_h, N_h)
        self.x_out_q = nn.Linear(N_h, N_h)
        self.x_out_c = nn.Linear(N_h, N_h)
        self.x_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        # Y col部分
        self.y_att = nn.Linear(N_h, N_h)
        self.y_out_q = nn.Linear(N_h, N_h)
        self.y_out_c = nn.Linear(N_h, N_h)
        self.y_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax(dim=1)

        self.CE = nn.CrossEntropyLoss()
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, col_emb_var, col_len, col_name_len, gt_set):
        max_q_len = max(q_len)
        max_col_len = max(col_len)

        B = len(q_len)

        # todo:encode sql query
        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)

        col_enc, _ = col_name_encode(self.col_lstm, col_emb_var, col_name_len, col_len)

        att_val_qc = torch.bmm(col_enc, self.type_att(q_enc).transpose(1,2))

        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, :, num:] = -100
        att_prob_qc = self.softmax(att_val_qc.view((-1, max_q_len))).view(B, -1, max_q_len)

        q_weighted = (q_enc.unsqueeze(1) * att_prob_qc.unsqueeze(3)).sum(2)

        type_out_temp = self.type_out_q(q_weighted) + self.type_out_c(col_enc)
        type_score = self.type_out(type_out_temp).squeeze()

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                type_score[idx, num:] = -100

        score = (type_score, 1)

        return score