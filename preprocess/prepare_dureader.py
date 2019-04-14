#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
对 dureader 数据集进行预处理，构建词典，构建词向量矩阵，并保存处理后的结果

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/3/31 15:49
"""
import sys

sys.path.append('../')

import os
import pickle
import yaml
from pprint import pprint
from utils.dataset import BRCDataset
from utils.vocab import Vocab


def prepare_dureader(input_dir,
                     data_cache_dir,
                     data_type,
                     max_p_num,
                     max_p_len,
                     max_q_len,
                     min_word_cnt,
                     embed_size,
                     embeddings_file,
                     train_badcase_save_file,
                     dev_badcase_save_file,
                     test_badcase_save_file,
                     stopwords=[],
                     word_translate_pipeline=[],
                     info=''):
    """
    :param embed_size: 词向量维度
    :param min_word_cnt: vocab 过滤的阈值
    :param max_q_len: 问题的最大长度
    :param max_p_len: doc 的最大长度
    :param max_p_num: 最大 doc 的数量
    :param input_dir: para_extracted 处理后的训练集、验证集合测试集的地址
    :param data_cache_dir: 保存可训练的训练集、验证集和测试集，以及对于的字典和词向量矩阵, feed_into_model
    :param data_type: 数据类型，search，zhidao
    :param embeddings_file: 预训练词向量的地址
    :param stopwords: 停用词列表
    :param word_translate_pipeline: 加载词向量时寻找词形式
    :param info: 备注信息
    """
    params = f'max_p_num{max_p_num}_max_p_len{max_p_len}_max_q_len{max_q_len}_min_word_cnt{min_word_cnt}_info{info}'

    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    for target_f in [os.path.join(data_cache_dir, d) for d in ['trainset', 'devset', 'testset', 'vocab']]:
        if not os.path.exists(target_f):
            os.makedirs(target_f)

    train_file = os.path.join(input_dir, 'trainset', '{}.train.json'.format(data_type))
    dev_file = os.path.join(input_dir, 'devset', '{}.dev.json'.format(data_type))
    test_file = os.path.join(input_dir, 'testset', '{}.test1.json'.format(data_type))

    print("=" * 10, 'load datas, build vocabulary and embeddings, Create BRCDataset', '=' * 10)
    print("* Create and save train brc dataset")
    train_brc_dataset = BRCDataset(max_p_num=max_p_num,
                                   max_p_len=max_p_len,
                                   max_q_len=max_q_len,
                                   train_files=[train_file],
                                   badcase_sample_log_file=train_badcase_save_file)
    print('* Building vocabulary...')
    vocab = Vocab()
    for word in train_brc_dataset.word_iter('train'):  # 根据 train 构建词典
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=min_word_cnt)
    filtered_num = unfiltered_vocab_size - vocab.size()
    print(f'Original vocab size: {unfiltered_vocab_size}')
    print(f'filter word_cnt<{min_word_cnt}: {filtered_num}')
    print(f'Final vocab size is {vocab.size()}')

    # vocab.randomly_init_embeddings(embed_size)
    vocab.build_embedding_matrix(embeddings_file)

    print('* Saving vocab...')
    with open(os.path.join(data_cache_dir, 'vocab', f'{data_type}.vocab.{params}.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    # train_brc_dataset.convert_to_ids(vocab)
    #
    # with open(os.path.join(data_cache_dir, "trainset", f"{data_type}.train_brcdataset.{params}.pkl"), 'wb') as pkl_file:
    #     pickle.dump(train_brc_dataset, pkl_file)
    #
    # print("* Create and save dev brc dataset")
    # valid_brc_dataset = BRCDataset(max_p_num=max_p_num,
    #                                max_p_len=max_p_len,
    #                                max_q_len=max_q_len,
    #                                dev_files=[dev_file],
    #                                badcase_sample_log_file=dev_badcase_save_file)
    #
    # valid_brc_dataset.convert_to_ids(vocab)
    #
    # with open(os.path.join(data_cache_dir, "devset", f"{data_type}.dev_brcdataset.{params}.pkl"), 'wb') as pkl_file:
    #     pickle.dump(valid_brc_dataset, pkl_file)
    #
    # print("* Create and save test brc dataset")
    # test_brc_dataset = BRCDataset(max_p_num=max_p_num,
    #                               max_p_len=max_p_len,
    #                               max_q_len=max_q_len,
    #                               test_files=[test_file],
    #                               badcase_sample_log_file=test_badcase_save_file)
    # test_brc_dataset.convert_to_ids(vocab)
    #
    # with open(os.path.join(data_cache_dir, "testset", f"{data_type}.test_brcdataset.{params}.pkl"), 'wb') as pkl_file:
    #     pickle.dump(test_brc_dataset, pkl_file)
    #
    # print('Done with preparing!')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess the DuReader dataset')
    parser.add_argument('--config',
                        default="../config/preprocess_config.yaml",
                        help='Path to a configuration file for preprocessing dureader data')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as cfg_file:
        config = yaml.load(cfg_file)

    print("=" * 20, 'preprocess configuration', '=' * 20)
    pprint(config)
    print()

    prepare_dureader(input_dir=os.path.normpath(config["mrc_dataset"]),
                     data_cache_dir=os.path.normpath(config["mrc_data_cache"]),
                     data_type=config['data_type'],
                     max_p_num=config['max_p_num'],
                     max_p_len=config['max_p_len'],
                     max_q_len=config['max_q_len'],
                     embeddings_file=os.path.normpath(config["embeddings_file"]),
                     train_badcase_save_file=config['bad_case_checking']['train_badcase_save_file'],
                     dev_badcase_save_file=config['bad_case_checking']['dev_badcase_save_file'],
                     test_badcase_save_file=config['bad_case_checking']['test_badcase_save_file'],
                     min_word_cnt=config['min_word_cnt'],
                     embed_size=config['embed_size'],
                     info=config["info"])
