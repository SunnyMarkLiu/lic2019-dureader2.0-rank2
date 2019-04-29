#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/20 19:28
"""
import sys
sys.path.append('../')

import json
from tqdm import tqdm
from utils.metric_util import read_data_to_dict, compute_bleu_rouge


def calc_one_sample_metric(sample):
    """ 计算 V1 数据一个样本的 rouge-l 和 bleu4 分数 """
    if len(sample['best_match_scores']) == 0:   # bad case
        return -1, -1

    pred_answers, ref_answers = [], []
    pred_answers.append({'question_id': sample['question_id'],
                         'question_type': sample['question_type'],
                         # 取 gold fake answer 作为预测的答案
                         'answers': [''.join(sample['fake_answers'][sample['best_match_scores'].index(max(sample['best_match_scores']))])],
                         'entity_answers': [[]],
                         'yesno_answers': []})
    ref_answers.append({'question_id': sample['question_id'],
                        'question_type': sample['question_type'],
                        'segmented_question': sample['segmented_question'],
                        'answers': [''.join(seg_ans) for seg_ans in sample['segmented_answers']],
                        'entity_answers': [[]],
                        'yesno_answers': [],
                        'documents': sample['documents']})

    pred_dict = read_data_to_dict(pred_answers)
    ref_dict = read_data_to_dict(ref_answers, is_ref=True)

    metrics = compute_bleu_rouge(pred_dict, ref_dict)
    rouge_l, bleu4 = metrics['ROUGE-L'], metrics['BLEU-4']
    return rouge_l, bleu4


check_dataset = {

    # ---------- baidu preprocess --------------
    # 'baidu_train': {
    #     'search': f'../input/dureader_baidu_preprocess_v0/mrc_dataset/trainset/search.train.json',
    #     'zhidao': f'../input/dureader_baidu_preprocess_v0/mrc_dataset/trainset/zhidao.train.json'
    # },
    #
    # 'baidu_dev': {
    #     'search': f'../input/dureader_baidu_preprocess_v0/mrc_dataset/devset/search.dev.json',
    #     'zhidao': f'../input/dureader_baidu_preprocess_v0/mrc_dataset/devset/zhidao.dev.json'
    # },

    # ------------ dureader_2.0_v4 -------------
    'train': {
        'search': f'../input/dureader_2.0_v4/mrc_dataset/trainset/search.train.json',
        'zhidao': f'../input/dureader_2.0_v4/mrc_dataset/trainset/zhidao.train.json'
    },

    # 'aug_train': {
    #     'search': f'../input/dureader_2.0_v4/mrc_dataset/aug_trainset/search.train.json',
    #     'zhidao': f'../input/dureader_2.0_v4/mrc_dataset/aug_trainset/zhidao.train.json'
    # },

    'dev': {
        'search': f'../input/dureader_2.0_v4/mrc_dataset/devset/search.dev.json',
        'zhidao': f'../input/dureader_2.0_v4/mrc_dataset/devset/zhidao.dev.json'
    },

    'cleaned18_dev': {
        'search': f'../input/dureader_2.0_v4/mrc_dataset/devset/cleaned_18.search.dev.json',
        'zhidao': f'../input/dureader_2.0_v4/mrc_dataset/devset/cleaned_18.zhidao.dev.json'
    }
}

# V3开始生成mrcdataset的时候就计算了 ceil rouge-l 和 bleu，直接统计即可
for data_type in check_dataset.keys():
    print(f"================== dureader_2.0_v4 {data_type} ceiling results ==================")
    for search_zhidao in check_dataset[data_type].keys():
        print(f"{search_zhidao}:")
        all_rouge_l, all_bleu4 = [], []
        with open(check_dataset[data_type][search_zhidao], 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                if not line.startswith('{'):
                    continue
                try:
                    sample = json.loads(line.strip())
                except:
                    continue
                rouge_l, bleu4 = sample['ceil_rouge_l'], sample['ceil_bleu4']
                if rouge_l > -1:
                    all_rouge_l.append(rouge_l)
                if bleu4 > -1:
                    all_bleu4.append(bleu4)

        print('mean rouge_l:', sum(all_rouge_l) / len(all_rouge_l))
        print('mean bleu4:', sum(all_bleu4) / len(all_bleu4))
        print()
