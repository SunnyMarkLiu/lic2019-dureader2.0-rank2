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
from utils.config_util import init_logging, read_config
from utils.dataset import Dataset
from utils.vocab import Vocab
from models import BiDAF, RNet

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

    experiment_params = 'model-{}_seed{}_data{}_max_p_num{}_max_p_len{}_max_q_len{}_min_word_cnt{}_preembedfile_{}'.format(
        global_config['global']['model'],
        global_config['global']['random_seed'],
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

    logger.info('reading dureader dataset')
    logging.info('create train BRCDataset')
    train_brc_dataset = Dataset(max_p_num=global_config['data']['max_p_num'],
                                max_p_len=global_config['data']['max_p_len'],
                                max_q_len=global_config['data']['max_q_len'],
                                train_files=[global_config['data']['mrc_dataset']['train_path']],
                                badcase_sample_log_file=global_config['data']['train_badcase_save_file'])

    vocab_path = os.path.join(global_config['data']['data_cache_dir'], 'vocab',
                              f"{global_config['data']['data_type']}.vocab.{experiment_params}.data")

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

    train_brc_dataset.convert_to_ids(vocab)

    model_choose = global_config['global']['model']
    logger.info(f"create {model_choose} model")

    if model_choose == 'bidaf':
        model = BiDAF()
    elif model_choose == 'rnet':
        model = RNet()
    else:
        raise ValueError('model "%s" in config file not recoginized' % model_choose)

    model = model.to(device)


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="train on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    args = parser.parse_args()

    train(args.config_path)
