# -*- coding: UTF-8 -*-
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from scripts.net_utils import run_lstm, col_name_encode


class CondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(CondPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.max_col_num = 45
        self.max_tok_num = 200

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word + N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(N_h, N_h)
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6)) # num of cols: 0-5

        self.q_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.op_att = nn.Linear(N_h, N_h)
        self.op_out_q = nn.Linear(N_h, N_h)
        self.op_out_c = nn.Linear(N_h, N_h)
        self.op_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 12)) #to 5

        self.cond_str_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h // 2,
                                     num_layers=N_depth, batch_first=True,
                                     dropout=0.3, bidirectional=True)
        self.cond_str_decoder = nn.LSTM(input_size=self.max_tok_num,
                                        hidden_size=N_h, num_layers=N_depth,
                                        batch_first=True, dropout=0.3)
        self.cond_str_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h // 2,
                                         num_layers=N_depth, batch_first=True,
                                         dropout=0.3, bidirectional=True)
        self.cond_str_out_g = nn.Linear(N_h, N_h)
        self.cond_str_out_h = nn.Linear(N_h, N_h)
        self.cond_str_out_col = nn.Linear(N_h, N_h)
        self.cond_str_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, col_emb_var, col_len, col_num, col_name_len, db_emb, gt_cond, gt_where, has_value):
        max_q_len = max(q_len)
        max_col_len = max(col_len)
        B = len(q_len)

        # q_emb_concat = torch.cat((q_emb_var, x_type_emb_var), 2)
        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        # col_enc, _ = col_name_encode(self.col_lstm, col_emb_var, col_name_len, col_len)

        col_emb_concat = torch.cat((col_emb_var, db_emb), 2)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_concat, col_len)

        # Predict column number: 0-4
        # att_val_qc_num: (B, max_col_len, max_q_len)
        att_val_qc_num = torch.bmm(col_enc, self.q_num_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_val_qc_num[idx, num:, :] = -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc_num[idx, :, num:] = -100
        att_prob_qc_num = self.softmax(att_val_qc_num.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted_num: (B, hid_dim)
        q_weighted_num = (q_enc.unsqueeze(1) * att_prob_qc_num.unsqueeze(3)).sum(2).sum(1)
        # self.col_num_out: (B, 4)
        col_num_score = self.col_num_out(self.col_num_out_q(q_weighted_num))

        # Predict columns.
        att_val_qc = torch.bmm(col_enc, self.q_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, :, num:] = -100
        att_prob_qc = self.softmax(att_val_qc.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted: (B, max_col_len, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qc.unsqueeze(3)).sum(2)
        # Compute prediction scores
        # self.col_out.squeeze(): (B, max_col_len)
        col_score = self.col_out(self.col_out_q(q_weighted) + self.col_out_c(col_enc)).squeeze()
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                col_score[idx, num:] = -100
        # get select columns for op prediction
        chosen_col_gt = []
        if gt_cond is None:
            cond_nums = np.argmax(col_num_score.data.cpu().numpy(), axis=1)
            col_scores = col_score.data.cpu().numpy()
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]]) for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [[x[0] for x in one_gt_cond] for one_gt_cond in gt_cond]

        col_emb = []
        for b in range(B):
            # stack函数, 叠加多个tensor的元素, 形成新的tensor,建立一个新的维度，然后再在该纬度上进行拼接
            # 设定的最多的条件数为 5 , 因此cur_col_emb长度需要补齐5
            cur_col_emb = torch.stack([col_enc[b, x]
                for x in chosen_col_gt[b]] + [col_enc[b, 0]] * (5 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        # Predict op
        op_att_val = torch.matmul(self.op_att(q_enc).unsqueeze(1),
                col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                op_att_val[idx, :, num:] = -100
        op_att = self.softmax(op_att_val.view(-1, max_q_len)).view(B, -1, max_q_len)
        q_weighted_op = (q_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)

        op_score = self.op_out(self.op_out_q(q_weighted_op) +
                            self.op_out_c(col_emb)).squeeze()

        # Predict the string of conditions
        # h_str_enc, _ = run_lstm(self.cond_str_lstm, q_emb_var, q_len)
        # e_cond_col, _ = col_name_encode(self.cond_str_name_enc, col_emb_var, col_name_len, col_len)
        cond_str_score = None
        if has_value:
            h_str_enc = q_enc
            e_cond_col = col_enc

            col_emb = []
            for b in range(B):
                cur_col_emb = torch.stack([e_cond_col[b, x]
                                           for x in chosen_col_gt[b]] +
                                          [e_cond_col[b, 0]] * (5 - len(chosen_col_gt[b])))
                col_emb.append(cur_col_emb)
            col_emb = torch.stack(col_emb)

            if gt_where is not None:
                gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where)
                g_str_s_flat, _ = self.cond_str_decoder(
                    gt_tok_seq.view(B * 5, -1, self.max_tok_num))
                g_str_s = g_str_s_flat.contiguous().view(B, 5, -1, self.N_h)

                h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
                g_ext = g_str_s.unsqueeze(3)
                col_ext = col_emb.unsqueeze(2).unsqueeze(2)

                cond_str_score = self.cond_str_out(
                    self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext) +
                    self.cond_str_out_col(col_ext)).squeeze()
                for b, num in enumerate(q_len):
                    if num < max_q_len:
                        cond_str_score[b, :, :, num:] = -100
            else:
                h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
                col_ext = col_emb.unsqueeze(2).unsqueeze(2)
                scores = []

                t = 0
                init_inp = np.zeros((B * 5, 1, self.max_tok_num), dtype=np.float32)
                init_inp[:, 0, 0] = 1  # Set the <BEG> token
                if self.gpu:
                    cur_inp = Variable(torch.from_numpy(init_inp).cuda())
                else:
                    cur_inp = Variable(torch.from_numpy(init_inp))
                cur_h = None
                # t代表value的最大长度
                while t < 10:
                    if cur_h:
                        g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp, cur_h)
                    else:
                        g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp)
                    g_str_s = g_str_s_flat.view(B, 5, 1, self.N_h)
                    g_ext = g_str_s.unsqueeze(3)

                    cur_cond_str_score = self.cond_str_out(
                        self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                        + self.cond_str_out_col(col_ext)).squeeze()
                    for b, num in enumerate(q_len):
                        if num < max_q_len:
                            cur_cond_str_score[b, :, num:] = -100
                    scores.append(cur_cond_str_score)

                    _, ans_tok_var = cur_cond_str_score.view(B * 5, max_q_len).max(1)
                    ans_tok = ans_tok_var.data.cpu()
                    data = torch.zeros(B * 5, self.max_tok_num).scatter_(
                        1, ans_tok.unsqueeze(1), 1)
                    if self.gpu:  # To one-hot
                        cur_inp = Variable(data.cuda())
                    else:
                        cur_inp = Variable(data)
                    cur_inp = cur_inp.unsqueeze(1)

                    t += 1

                cond_str_score = torch.stack(scores, 2)
                for b, num in enumerate(q_len):
                    if num < max_q_len:
                        cond_str_score[b, :, :, num:] = -100  # [B, IDX, T, TOK_NUM]
        score = (col_num_score, col_score, op_score, cond_str_score)

        return score

    def gen_gt_batch(self, split_tok_seq):
        B = len(split_tok_seq)
        max_len = max([max([len(tok) for tok in tok_seq] + [0]) for
                       tok_seq in split_tok_seq]) - 1  # The max seq len in the batch.
        if max_len < 1:
            max_len = 1
        ret_array = np.zeros((
            B, 5, max_len, self.max_tok_num), dtype=np.float32)
        ret_len = np.zeros((B, 5))
        for b, tok_seq in enumerate(split_tok_seq):
            idx = 0
            for idx, one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1]
                ret_len[b, idx] = len(out_one_tok_seq)
                for t, tok_id in enumerate(out_one_tok_seq):
                    ret_array[b, idx, t, tok_id] = 1
            if idx < 4:
                ret_array[b, idx + 1:, 0, 1] = 1
                ret_len[b, idx + 1:] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp)

        return ret_inp_var, ret_len  # [B, IDX, max_len, max_tok_num]
