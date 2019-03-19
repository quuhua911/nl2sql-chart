# -*- coding: UTF-8 -*-
import torch
import datetime
import logging
import numpy as np
from scripts.utils import *
from scripts.model.nlnet.nlnet import NLNet

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/train.log',
                    filemode='w')

if __name__ == '__main__':
    N_word = 300
    B_word = 42
    TEST = True

    if TEST:
        FAST = True
        GPU = False
        BATCH_SIZE = 20
    else:
        FAST = False
        GPU = True
        BATCH_SIZE = 64

    TRAIN_ENTRY = (True, True, True)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY

    learning_rate = 1e-3

    file_dir = "data/"

    sql_data, table_data, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, schemas, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(file_dir, FAST=FAST)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), FAST=FAST)

    model = NLNet(word_emb, N_word=N_word, gpu=GPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, schemas, TRAIN_ENTRY)

    best_sel_acc = init_acc[1][0]
    best_cond_acc = init_acc[1][1]
    best_group_acc = init_acc[1][2]
    best_order_acc = init_acc[1][3]
    best_tot_acc = 0.0

    for i in range(300):
        logging.info('Epoch %d @ %s' % (i+1, datetime.datetime.now()))

        loss = epoch_train(model, optimizer, BATCH_SIZE, sql_data, table_data, schemas, TRAIN_ENTRY)
        logging.info('Loss = %s' % loss)

        train_tot_acc, train_par_acc = epoch_acc(model, BATCH_SIZE, sql_data, table_data, schemas, TRAIN_ENTRY)
        logging.info('Train acc_qm: %s' % train_tot_acc)

        logging.info('Train parts acc: sel: %s, cond: %s, group: %s, order: %s' % (train_par_acc[0], train_par_acc[1], train_par_acc[2], train_par_acc[3]))

        val_tot_acc, val_par_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, schemas, TRAIN_ENTRY, error_print=False)
        logging.info('Vak acc_qm: %s' % val_tot_acc)

        logging.info('Val parts acc: sel: %s, cond: %s, group: %s, order: %s' % (val_par_acc[0], val_par_acc[1], val_par_acc[2], val_par_acc[3]))

        if val_par_acc[0] > best_sel_acc:
            best_sel_acc = val_par_acc[0]
            logging.info("Saving sel model...")
            torch.save(model.sel_pred.state_dict(), "test_saved_models/sel_models.dump")
        if val_par_acc[1] > best_cond_acc:
            best_cond_acc = val_par_acc[1]
            logging.info("Saving cond model...")
            torch.save(model.cond_pred.state_dict(), "test_saved_models/cond_models.dump")
        if val_par_acc[2] > best_group_acc:
            best_group_acc = val_par_acc[2]
            logging.info("Saving group model...")
            torch.save(model.group_pred.state_dict(), "test_saved_models/group_models.dump")
        if val_par_acc[3] > best_order_acc:
            best_order_acc = val_par_acc[3]
            logging.info("Saving order model...")
            torch.save(model.order_pred.state_dict(), "test_saved_models/order_models.dump")
        if val_tot_acc > best_tot_acc:
            best_tot_acc = val_tot_acc

        logging.info(' Best val sel = %s, cond = %s, group = %s, order = %s, tot = %s' % (best_sel_acc, best_cond_acc, best_group_acc, best_order_acc, best_tot_acc))




