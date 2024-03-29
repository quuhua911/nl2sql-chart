# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu):
        super(WordEmbedding, self).__init__()

        self.N_word = N_word
        self.gpu = gpu

        self.word_emb = word_emb

    # 处理自然语言的q， 获得对应的Embedding,此时已包括<beg>和<end>
    def gen_x_batch(self, q, col, is_list=False, is_q=False, is_col=False):
        B = len(q)
        val_embs = []

        # 用于记录每个query有几个词
        val_len = np.zeros(B, dtype=np.int64)

        '''
        zip函数
        zip函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表

        a>[1,2,3]
        b>[3,2,1,0]
        zip(a,b) => [(1,3),(2,2),(3,1)] 元素个数与最短的列表一致
        '''

        temp = zip(q, col)
        for i, (one_q, one_col) in enumerate(temp):
            if not is_list:
                q_val = map(lambda x: self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q)
            else:
                q_val = []
                # 句粒度
                for ws in one_q:
                    emb_list = []
                    if type(ws).__name__ == 'int':
                        print(one_q)
                    ws_len = len(ws)
                    # 词粒度
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                    if ws_len == 0:
                        raise Exception("word list shouldn't be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))
                        '''
                        # 取嵌入的均值
                        emb_list_without_map = []
                        for emb in emb_list:
                            emb_from_ndarray = emb.tolist()
                            temp_type = type(emb_from_ndarray).__name__
                            if temp_type == 'map':
                                the_list = list(emb_from_ndarray)
                                if len(the_list) == 0:
                                    emb = np.zeros(self.N_word, dtype=np.float32)
                                else:
                                    emb = np.array(the_list)
                            emb_list_without_map.append(emb)
                        q_val.append(sum(emb_list_without_map) / float(ws_len))
                        '''
                        '''
                        emb_list_without_map = []
                        for emb in emb_list:
                            emb_from_ndarray = emb.tolist()
                            temp = []
                            for item in emb_from_ndarray:
                                temp.append(item)

                            # todo : 遍历map后迭代器内容为空
                            if len(temp) != 0:
                                emb_list_without_map.append(temp)

                        emb_sum = np.sum(emb_list_without_map, axis=0)
                        theType = type(emb_sum).__name__
                        if theType == 'float64':
                            emb_sum = np.zeros(self.N_word, dtype=np.float32).tolist()
                        emb_avg = list(map(lambda x: x/float(ws_len), emb_sum))
                        q_val.append(emb_avg)
                        '''
            if not is_list or is_q:
                q_val_list = list(q_val)
                '''
                for t in range(len(q_val_list)):
                    temp = q_val_list[t].tolist()
                    if type(temp).__name__ == 'map':
                        temp = list(temp)
                    if len(temp) != 0:
                        content = np.array(temp)
                    else:
                        content = np.zeros(self.N_word)
                    q_val_list[t] = content
                '''
                if not is_col:
                    val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val_list + [
                        np.zeros(self.N_word, dtype=np.float32)])  # <BEG> and <END>
                    val_len[i] = 1 + len(q_val_list) + 1
                else:
                    val_embs.append(q_val_list)
                    val_len[i] = len(q_val_list)
            else:
                q_val_list = list(q_val)
                val_embs.append(q_val_list)
                val_len[i] = len(q_val_list)

        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)

        # 补齐len至max_len
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]

        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        # Variable是对Tensor的封装
        val_inp_var = Variable(val_inp)
        # val_inp shape (sql数目, query最长长度, 词嵌入长度)
        return val_inp_var, val_len

    # 处理列表名获得词嵌入
    # col代表每个输入所涉及到字段组，是个list
    def gen_col_batch(self, col):
        ret = []

        # 用于记录每个输入所涉及到的字段数目
        col_len = np.zeros(len(col), dtype=np.int64)

        names = []
        # one_col为某一个输入对应的所有字段名,是个list
        for i, one_col in enumerate(col):
            names = names + one_col
            col_len[i] = len(one_col)

        # 最终的names是所有字段名的集合, 包括重复的字段
        name_inp_var, name_len = self.str_list_to_batch(names)

        # name_len: 由各个字段词数组成的列表
        # col_len: 由各个输入对应的表的字段数组成的列表
        return name_inp_var, name_len, col_len

    # detailed info
    def str_list_to_batch(self, col_names):
        B = len(col_names)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)

        # preprocess for embedding
        for i, one_str in enumerate(col_names):
            val = [self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_str]

            val_embs.append(val)
            val_len[i] = len(val)

        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)

        # 每个col_name
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len


