#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/14 20:57
"""
from util.dataset import Dataset
import logging
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
               badcase_sample_log_file='badcase_sample_log_file.json')
batch_size = 32

for i in range(10):
    real_batch_size = data.get_real_batch_size(batch_size=batch_size, set_name='train')
    print('real batch size after bin cut: {}'.format(real_batch_size))

    total_batch_count = 0
    train_batches = data.gen_mini_batches('train', batch_size, 0, shuffle=True, calc_total_batch_cnt=True)
    for batch in train_batches:
        if len(batch) == real_batch_size:
            total_batch_count += 1

    print('total training batch counts: {}'.format(total_batch_count))

    train_batches = data.gen_mini_batches('train', batch_size, 0, shuffle=True)

    tqdm_batch_iterator = tqdm(train_batches, total=total_batch_count)
    for bitx, batch in enumerate(tqdm_batch_iterator):
        pass
