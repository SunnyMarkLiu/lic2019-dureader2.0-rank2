#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
验证生成的 MRC 数据集的有效性

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/12 17:08
"""
import sys
import os
import json
from tqdm import tqdm

# python check_ceil_rouge_badcase.py 50 50 ../input/dureader_2.0_v4/final_mrc_dataset/trainset/search.train.json
# python check_ceil_rouge_badcase.py 60 60 ../input/dureader_2.0_v4/final_mrc_dataset/trainset/zhidao.train.json
if __name__ == '__main__':
    # dureader_2.0 / dureader_2.0_v4
    bad_case_rouge_l_threshold = float(sys.argv[1])    # 0
    bad_case_bleu_threshold = float(sys.argv[2])    # 0
    dataset_file = sys.argv[3]

    data_version = dataset_file.split('/')[2]

    if 'zhidao' in dataset_file:
        search_zhidao = 'zhidao'
    else:
        search_zhidao = 'search'

    if 'train' in dataset_file:
        data_type = 'train'
    elif 'cleaned_18' in dataset_file:
        data_type = 'cleaned_18_dev'
    elif 'dev' in dataset_file:
        data_type = 'dev'
    else:
        raise ValueError('error data_type, must be train/dev/cleaned_18')

    if not os.path.exists(f'./bad_case_train_dev_sample/{data_version}/'):
        os.mkdir(f'./bad_case_train_dev_sample/{data_version}/')

    bad_case_writer = open(f'./bad_case_train_dev_sample/{data_version}/{data_type}_{search_zhidao}_bad_case_sample_rougel_lessthan{bad_case_rouge_l_threshold}.json', 'w')
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if not line.startswith('{'):
                continue

            sample = json.loads(line.strip())
            rouge_l, bleu4 = sample['ceil_rouge_l'], sample['ceil_bleu4']

            if (rouge_l <= bad_case_rouge_l_threshold) and (bleu4 <= bad_case_bleu_threshold):
                bad_case_writer.write(line)
    bad_case_writer.flush()
    bad_case_writer.close()
