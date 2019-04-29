#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/13 14:35
"""
import os
import logging
import argparse
import pickle
import torch
import random
import numpy as np
from pprint import pprint
from utils.config_util import init_logging, read_config
from utils.dataset import Dataset
from utils.vocab import Vocab
import torch.optim as optim
from torchmrc.models import MatchLSTM
from torchmrc.modules.loss import MyNLLLoss
logger = logging.getLogger(__name__)


def seed_torch(random_seed=1):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config_path):
    logger.info('------------MODEL TRAIN--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)
    pprint(global_config)
    print()

    experiment_params = '{}_max_p_num{}_max_p_len{}_max_q_len{}_min_word_cnt{}_preembedfile_{}'.format(
        global_config['data']['data_type'],
        global_config['data']['max_p_num'],
        global_config['data']['max_p_len'],
        global_config['data']['max_q_len'],
        global_config['data']['min_word_cnt'],
        global_config['data']['embeddings_file'].split('/')[-1]
    )

    # seed everything for torch
    seed_torch(global_config['global']['random_seed'])
    device = torch.device("cuda:{}".format(global_config['global']['gpu']) if torch.cuda.is_available() else "cpu")

    # -------------------- Data preparing -------------------
    logger.info('reading dureader dataset [{}]'.format(global_config['data']['data_type']))
    logging.info('create train BRCDataset')
    train_badcase_save_file = '{}/{}/train_badcase.log'.format(global_config['data']['train_badcase_save_path'],
                                                               global_config['data']['data_type'],)
    train_brc_dataset = Dataset(max_p_num=global_config['data']['max_p_num'],
                                max_p_len=global_config['data']['max_p_len'],
                                max_q_len=global_config['data']['max_q_len'],
                                train_files=[global_config['data']['mrc_dataset']['train_path']],
                                badcase_sample_log_file=train_badcase_save_file)

    logging.info(f"Building {global_config['data']['data_type']} vocabulary from train {global_config['data']['data_type']} text set")

    vocab_path = os.path.join(global_config['data']['data_cache_dir'], '{}.vocab'.format(experiment_params))
    if not os.path.exists(vocab_path):
        logging.info('Building vocabulary from train text set')
        vocab = Vocab()
        for word in train_brc_dataset.word_iter('train'):  # 根据 train 构建词典
            vocab.add(word)

        unfiltered_vocab_size = vocab.size()
        vocab.filter_tokens_by_cnt(min_cnt=global_config['data']['min_word_cnt'])
        filtered_num = unfiltered_vocab_size - vocab.size()
        logger.info(f'Original vocab size: {unfiltered_vocab_size}')
        logger.info(f"filter word_cnt<{global_config['data']['min_word_cnt']}: {filtered_num}, left: {vocab.size()}")

        logger.info('load pretrained embeddings and build embedding matrix')
        vocab.build_embedding_matrix(global_config['data']['embeddings_file'])

        logger.info('Saving vocab')
        with open(vocab_path, 'wb') as fout:
            pickle.dump(vocab, fout)
    else:
        logging.info(f'load vocabulary from {vocab_path}')
        with open(vocab_path, 'rb') as f:
            vocab: Vocab = pickle.load(f)
            logging.info(f'vocabulary size: {vocab.size()}')
            logging.info(f'trainable oov words start from 0 to {vocab.oov_word_end_idx}')

    logger.info('train brc dataset convert to ids')
    train_brc_dataset.convert_to_ids(vocab, use_oov2unk=True)

    model_choose = global_config['global']['model']
    logger.info(f"create {model_choose} model")

    # NOTE: embedding matrix should be float32, to avoid LSTM CUDNN_STATUS_BAD_PARAM error
    embedding_matrix = torch.from_numpy(np.array(vocab.embedding_matrix, dtype=np.float32)).to(device)

    # ----------------- build neural network model, loss func, optimizer, scheduler ------------------
    if model_choose == 'match_lstm':
        model = MatchLSTM(max_p_num=global_config['data']['max_p_num'],
                          max_p_len=global_config['data']['max_p_len'],
                          max_q_len=global_config['data']['max_q_len'],
                          vocab_size=vocab.size(),
                          embed_dim=vocab.embed_dim,
                          rnn_mode='LSTM',
                          hidden_size=100,
                          encoder_bidirection=True,
                          match_lstm_bidirection=True,
                          gated_attention=True,
                          rnn_dropout=0.1,
                          enable_layer_norm=False,
                          ptr_bidirection=False,
                          embed_matrix=embedding_matrix,
                          embed_trainable=False,
                          embed_bn=False,
                          padding_idx=vocab.get_id(vocab.pad_token),
                          embed_dropout=0,
                          device=device)
    # elif model_choose == 'rnet':
    #     model = RNet()
    else:
        raise ValueError('model "%s" in config file not recoginized' % model_choose)

    model = model.to(device)
    print('\n', model, '\n')

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = global_config['train']['learning_rate']
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_choose == 'adamax':
        optimizer = optim.Adamax(optimizer_param)
    elif optimizer_choose == 'adadelta':
        optimizer = optim.Adadelta(optimizer_param)
    elif optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param)
    elif optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param, lr=optimizer_lr)
    else:
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    # loss function
    criterion = MyNLLLoss()
    # training arguments
    logger.info('start training...')
    train_batch_size = global_config['train']['batch_size']
    valid_batch_size = global_config['train']['valid_batch_size']

    question = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 79],
                [1, 2, 3, 4, 5, 6, 7, 8, 79],
                [1, 2, 3, 4, 5, 6, 79, 79, 79],
                [1, 2, 3, 4, 5, 6, 79, 79, 79],

                [11, 22, 33, 44, 55, 66, 77, 88, 99],
                [11, 22, 33, 44, 55, 66, 79, 79, 79],
                [11, 22, 33, 44, 55, 66, 77, 88, 79],
                [79] * 9,
                [79] * 9
                ]
    question = torch.tensor(question, dtype=torch.long)

    context = [[110, 210, 310, 410, 510, 610, 720, 850, 920, 760, 820, 90],
               [120, 220, 320, 420, 520, 630, 730, 860, 930, 750, 840, 79],
               [130, 230, 340, 430, 530, 640, 749, 779, 749, 749, 759, 79],
               [140, 240, 330, 440, 540, 650, 750, 880, 950, 739, 769, 79],
               [140, 240, 330, 440, 540, 650, 750, 880, 950, 739, 769, 79],

               [11, 222, 332, 443, 554, 662, 776, 887, 969, 11, 22, 33],
               [111, 223, 333, 44, 55, 66, 77, 88, 99, 33, 33, 44],
               [11, 220, 133, 414, 575, 656, 747, 838, 929, 79, 79, 79],
               [79] * 12,
               [79] * 12
               ]
    context = torch.tensor(context, dtype=torch.long)
    model.eval()
    ans_range_prop, ans_range, vis_param= model.forward(question.to(device), context.to(device), passage_cnts=[5, 5])
    print()
    print(ans_range_prop)
    print(ans_range)
    # print(vis_param)

if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="train on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    args = parser.parse_args()

    train(args.config_path)
