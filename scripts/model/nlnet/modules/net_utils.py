# coding=UTF-8
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def run_lstm(lstm, inp, inp_len, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.

    # 根据query中词的数目降序排列，返回各个长度的index
    # ex. sort_perm [2 3 4 7 5 6 8 9 1 0] 表示　原数组第三位最长
    # array[sort_perm] 可获得排序后的序列
    sort_perm = np.array(sorted(range(len(inp_len)),
                                key=lambda k: inp_len[k], reverse=True))

    # 获得数目的降序排列
    sort_inp_len = inp_len[sort_perm]

    # argsort(list)，先把list进行排序，再返回各个值在list中idx,生成list,默认升序
    # sort_perm_inv [9 8 0 1 2 4 5 3 6 7] 表示 排序后的 list [0 1 2 3 4 5 6 7 8 9]中 第一位在原数组第十位
    # array[sort_perm_inv] 可获得 sort_perm
    sort_perm_inv = np.argsort(sort_perm)

    # pack_padded_sequence(input, lengths, batch_first)
    # input为已经排好序的数组,lengths为降序排列的长度的数组
    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
                                                 sort_inp_len, batch_first=True)
    # pack_padded_sequence 函数将 T*B*N 的 T*B 两维合并

    sort_ret_s, sort_ret_h = lstm(lstm_inp)

    # 恢复顺序
    ret_s = nn.utils.rnn.pad_packed_sequence(
        sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(lstm, name_inp_var, name_len, col_len):
    # Encode the columns.
    # The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(lstm, name_inp_var, name_len)
    temp = tuple(range(len(name_len)))
    name_out = name_hidden[temp, name_len - 1]
    ret = torch.FloatTensor(
        len(col_len), max(col_len), name_out.size()[1]).zero_()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st + cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len

