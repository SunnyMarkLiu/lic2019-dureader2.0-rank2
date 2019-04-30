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
from torchmrc.modules.loss import MRCStartEndNLLLoss
from utils.model_util import model_training
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

    # -------------------------- parameters -----------------------
    # global
    random_seed = global_config['global']['random_seed']
    gpu = global_config['global']['gpu']
    
    # data
    data_type = global_config['data']['data_type']
    max_p_num = global_config['data']['max_p_num']
    max_p_len = global_config['data']['max_p_len']
    max_q_len = global_config['data']['max_q_len']
    min_word_cnt = global_config['data']['min_word_cnt']

    train_path = global_config['data']['mrc_dataset']['train_path']
    
    # train
    learning_rate = global_config['train']['learning_rate']
    optimizer_choose = global_config['train']['optimizer']
    train_batch_size = global_config['train']['batch_size']
    valid_batch_size = global_config['train']['valid_batch_size']
    clip_grad_norm = global_config['train']['clip_grad_norm']

    # --------------------------------------------------------------

    experiment_params = '{}_max_p_num{}_max_p_len{}_max_q_len{}_min_word_cnt{}_preembedfile_{}'.format(
        data_type,
        max_p_num,
        max_p_len,
        max_q_len,
        min_word_cnt,
        global_config['data']['embeddings_file'].split('/')[-1]
    )

    # seed everything for torch
    seed_torch(random_seed)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    # -------------------- Data preparing -------------------
    logger.info('reading dureader dataset [{}]'.format(data_type))
    logging.info('create train BRCDataset')
    train_badcase_save_file = '{}/{}/train_badcase.log'.format(global_config['data']['train_badcase_save_path'],
                                                               data_type, )
    train_brc_dataset = Dataset(max_p_num=max_p_num,
                                max_p_len=max_p_len,
                                max_q_len=max_q_len,
                                is_train=True,
                                data_files=[train_path],
                                badcase_sample_log_file=train_badcase_save_file)

    logging.info(f"Building {data_type} vocabulary from train {data_type} text set")

    vocab_path = os.path.join(global_config['data']['data_cache_dir'], '{}.vocab'.format(experiment_params))
    if not os.path.exists(vocab_path):
        logging.info('Building vocabulary from train text set')
        vocab = Vocab()
        for word in train_brc_dataset.word_iter():  # 根据 train 构建词典
            vocab.add(word)

        unfiltered_vocab_size = vocab.size()
        vocab.filter_tokens_by_cnt(min_cnt=min_word_cnt)
        filtered_num = unfiltered_vocab_size - vocab.size()
        logger.info(f'Original vocab size: {unfiltered_vocab_size}')
        logger.info(f"filter word_cnt<{min_word_cnt}: {filtered_num}, left: {vocab.size()}")

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

    # --------- convert text to ids -----------
    logger.info('train brc dataset convert to ids')
    train_brc_dataset.update_pad_id(vocab.get_id(vocab.pad_token))  # set pad token id
    train_brc_dataset.convert_to_ids(vocab, use_oov2unk=True)

    # --------------- model prepare ---------------
    model_choose = global_config['global']['model']
    logger.info(f"create {model_choose} model")

    # NOTE: embedding matrix should be float32, to avoid LSTM CUDNN_STATUS_BAD_PARAM error
    embedding_matrix = torch.from_numpy(np.array(vocab.embedding_matrix, dtype=np.float32)).to(device)

    # ----------------- build neural network model, loss func, optimizer, scheduler ------------------
    if model_choose == 'match_lstm':
        model = MatchLSTM(max_p_num=max_p_num,
                          max_p_len=max_p_len,
                          max_q_len=max_q_len,
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
    logger.info(f'{model_choose} model structure:\n' + model.__str__() + '\n')

    # optimizer
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_choose == 'adamax':
        optimizer = optim.Adamax(optimizer_param, lr=learning_rate)
    elif optimizer_choose == 'adadelta':
        optimizer = optim.Adadelta(optimizer_param, lr=learning_rate)
    elif optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param, lr=learning_rate)
    elif optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param, lr=learning_rate)
    else:
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    # loss function
    criterion = MRCStartEndNLLLoss()

    # ---------------------- training and validate ----------------------
    logger.info('start training...')
    for i in range(10):
        model_training(model, optimizer, criterion, train_brc_dataset, train_batch_size, clip_grad_norm, device)

if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="train on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    args = parser.parse_args()

    train(args.config_path)
