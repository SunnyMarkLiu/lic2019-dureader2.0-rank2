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
    """ 计算一个样本的 rouge-l 和 bleu4 分数 """
    pred_answers, ref_answers = [], []
    pred_answers.append({'question_id': sample['question_id'],
                         'question_type': sample['question_type'],
                         'answers': [''.join(ans) for ans in sample['fake_answers']],
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


data_version = 'dureader_2.0'

all_rouge_l, all_bleu4 = [], []
print(f"================== {data_version} train ceiling results ==================")
print("search:")
with open(f'../input/{data_version}/mrc_dataset/trainset/search.train.json', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        sample = json.loads(line.strip())
        rouge_l, bleu4 = calc_one_sample_metric(sample)
        all_rouge_l.append(rouge_l)
        all_bleu4.append(bleu4)

print('mean rouge_l:', sum(all_rouge_l) / len(all_rouge_l))
print('mean bleu4:', sum(all_bleu4) / len(all_bleu4))
print()

print("zhidao:")
all_rouge_l, all_bleu4 = [], []
with open(f'../input/{data_version}/mrc_dataset/trainset/zhidao.train.json', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        sample = json.loads(line.strip())
        rouge_l, bleu4 = calc_one_sample_metric(sample)
        all_rouge_l.append(rouge_l)
        all_bleu4.append(bleu4)

print('mean rouge_l:', sum(all_rouge_l) / len(all_rouge_l))
print('mean bleu4:', sum(all_bleu4) / len(all_bleu4))

print(f"================== {data_version} dev ceiling results ==================")
print("search:")
all_rouge_l, all_bleu4 = [], []
with open(f'../input/{data_version}/mrc_dataset/devset/search.dev.json', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        sample = json.loads(line.strip())
        rouge_l, bleu4 = calc_one_sample_metric(sample)
        all_rouge_l.append(rouge_l)
        all_bleu4.append(bleu4)

print('mean rouge_l:', sum(all_rouge_l) / len(all_rouge_l))
print('mean bleu4:', sum(all_bleu4) / len(all_bleu4))
print()

print("zhidao:")
all_rouge_l, all_bleu4 = [], []
with open(f'../input/{data_version}/mrc_dataset/devset/zhidao.dev.json', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        sample = json.loads(line.strip())
        rouge_l, bleu4 = calc_one_sample_metric(sample)
        all_rouge_l.append(rouge_l)
        all_bleu4.append(bleu4)

print('mean rouge_l:', sum(all_rouge_l) / len(all_rouge_l))
print('mean bleu4:', sum(all_bleu4) / len(all_bleu4))

print(f"================== {data_version} cleaned18 dev ceiling results ==================")
print("search:")
all_rouge_l, all_bleu4 = [], []
with open(f'../input/{data_version}/mrc_dataset/devset/cleaned_18.search.dev.json', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        sample = json.loads(line.strip())
        rouge_l, bleu4 = calc_one_sample_metric(sample)
        all_rouge_l.append(rouge_l)
        all_bleu4.append(bleu4)

print('mean rouge_l:', sum(all_rouge_l) / len(all_rouge_l))
print('mean bleu4:', sum(all_bleu4) / len(all_bleu4))
print()

print("zhidao:")
all_rouge_l, all_bleu4 = [], []
with open(f'../input/{data_version}/mrc_dataset/devset/cleaned_18.zhidao.dev.json', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        sample = json.loads(line.strip())
        rouge_l, bleu4 = calc_one_sample_metric(sample)
        all_rouge_l.append(rouge_l)
        all_bleu4.append(bleu4)

print('mean rouge_l:', sum(all_rouge_l) / len(all_rouge_l))
print('mean bleu4:', sum(all_bleu4) / len(all_bleu4))
