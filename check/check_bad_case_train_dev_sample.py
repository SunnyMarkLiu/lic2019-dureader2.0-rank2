#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
验证生成的 MRC 数据集的有效性

empty fake answer：/
百度开源的 preprocessed 数据的 empty fake answer：13099/271570

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/12 17:08
"""
import sys
import json
from tqdm import tqdm

if __name__ == '__main__':
    # dureader_2.0 / dureader_2.0_v3
    data_version = sys.argv[1]
    bad_case_rouge_l_threshold = float(sys.argv[2])    # 0

    check_dataset = {
        'train': {
            'search': f'../input/{data_version}/mrc_dataset/trainset/search.train.json',
            'zhidao': f'../input/{data_version}/mrc_dataset/trainset/zhidao.train.json'
        },

        'dev': {
            'search': f'../input/{data_version}/mrc_dataset/devset/search.dev.json',
            'zhidao': f'../input/{data_version}/mrc_dataset/devset/zhidao.dev.json'
        },

        'cleaned18_dev': {
            'search': f'../input/{data_version}/mrc_dataset/devset/cleaned_18.search.dev.json',
            'zhidao': f'../input/{data_version}/mrc_dataset/devset/cleaned_18.zhidao.dev.json'
        }
    }

    for data_type in check_dataset.keys():
        print(f"================== {data_version} {data_type} ceiling results ==================")
        for search_zhidao in check_dataset[data_type].keys():
            print(f"{search_zhidao}:")
            bad_case_writer = open(f'./bad_case_train_dev_sample/{data_type}_{search_zhidao}_bad_case_sample_{data_version}_rougel_lessthan{bad_case_rouge_l_threshold}.json', 'w')
            with open(check_dataset[data_type][search_zhidao], 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    if not line.startswith('{'):
                        continue

                    sample = json.loads(line.strip())
                    rouge_l, bleu4 = sample['ceil_rouge_l'], sample['ceil_bleu4']

                    if rouge_l <= bad_case_rouge_l_threshold:
                        bad_case_writer.write(line + '\n')
            bad_case_writer.flush()
            bad_case_writer.close()
