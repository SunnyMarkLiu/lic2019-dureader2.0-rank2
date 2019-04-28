#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/28 08:29
"""
import sys

sys.path.append('../')

import json
from tqdm import tqdm
from utils.metric_util import read_data_to_dict, compute_bleu_rouge


def calc_one_sample_metric(sample):
    """ 计算一个样本的 rouge-l 和 bleu4 分数 """
    if len(sample['segmented_answers']) == 0 or len(sample['answers']) == 0:  # bad case
        return -1, -1

    pred_answers, ref_answers = [], []
    pred_answers.append({'question_id': sample['question_id'],
                         'question_type': sample['question_type'],
                         # 取 gold fake answer 作为预测的答案
                         'answers': sample['fake_answers'],
                         'entity_answers': [[]],
                         'yesno_answers': []})
    ref_answers.append({'question_id': sample['question_id'],
                        'question_type': sample['question_type'],
                        'segmented_question': sample['segmented_question'],
                        'answers': sample['answers'],
                        'entity_answers': [[]],
                        'yesno_answers': [],
                        'documents': sample['documents']})

    pred_dict = read_data_to_dict(pred_answers)
    ref_dict = read_data_to_dict(ref_answers, is_ref=True)

    metrics = compute_bleu_rouge(pred_dict, ref_dict)
    rouge_l, bleu4 = metrics['ROUGE-L'], metrics['BLEU-4']
    return rouge_l, bleu4


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        rouge_l, bleu4 = calc_one_sample_metric(sample)
        sample['ceil_rouge_l'], sample['ceil_bleu4'] = rouge_l, bleu4
        print(json.dumps(sample, ensure_ascii=False))
