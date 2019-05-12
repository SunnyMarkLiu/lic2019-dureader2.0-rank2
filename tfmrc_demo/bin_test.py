#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/12 12:58
"""
from model import MultiAnsModel
from util.dataset import Vocab
from util.dataset import Dataset
import logging
import pickle
import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

data = Dataset(5, 500, 20, 300,
               train_answer_len_cut_bins=6,
               train_files=['../input/dureader_2.0_v5/final_mrc_dataset/trainset/search.train.json'],
               cleaned18_dev_files=['../input/dureader_2.0_v5/final_mrc_dataset/devset/cleaned_18.search.dev.json'],
               badcase_sample_log_file='badcase_sample_log_file.json')
batch_size = 32
real_batch_size = data.get_real_batch_size(batch_size=batch_size, set_name='train')

print('real batch size after bin cut: {}'.format(real_batch_size))

# 对于少于 batch_size 的 batch 数据进行丢弃，所以此处不加 1
total_batch_count = data.get_data_length('train') // real_batch_size
train_batches = data.gen_mini_batches('train', batch_size, -1, shuffle=True)

tqdm_batch_iterator = tqdm(train_batches, total=total_batch_count)
for bitx, batch in enumerate(tqdm_batch_iterator):
    pass
print('done')
