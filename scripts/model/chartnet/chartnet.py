import torch
import torch.nn as nn
class chartNet(nn.module):
    def __init__(self, word_emb, N_word, N_h=120, N_depth=2, gpu=False):
        super(chartNet, self).__init__()

        self.gpu = gpu