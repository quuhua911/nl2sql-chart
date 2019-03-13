import torch
import torch.nn as nn
from scripts.word_embedding import WordEmbedding
class chartNet(nn.module):
    def __init__(self, word_emb, N_word, N_h=120, N_depth=2, gpu=False):
        super(chartNet, self).__init__()

        self.gpu = gpu
        self.N_word = N_word
        self.N_h = N_h
        self.N_depth =N_depth

        self.embed_layer = WordEmbedding(word_emb, N_word, gpu)

        self.chart_pred = ChartPredictor(N_word, N_h, N_depth)

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    def forward(self, q, col, col_num, pred_entry, gt_cond=None, gt_sel=None):
        B = len(q)

        x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
        col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)

        chart_score = self.chart_pred(x_emb_var, x_len, col_inp_var, col_len, col_name_len, gt_sel)

        return chart_score

    def loss(self, score, truth):

        type_score, x_col_score, y_col_score = score

        B = len(truth)

        loss = 0
