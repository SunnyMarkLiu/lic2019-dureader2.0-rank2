#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/5 16:49
"""
import sys

sys.path.append('../')

import sys
import json
import itertools
from utils.metric_util import metric_max_over_ground_truths, f1_score
import warnings

warnings.filterwarnings("ignore")


def split_list_by_specific_value(iterable, splitters):
    return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]


def contain_sublist(passage, answer):
    start = -1
    end = -1
    for i in range(len(passage)):
        if passage[i] == answer[0]:
            start = i
            right = 1
            while (i + right < len(passage)) and (right < len(answer)) and (passage[i + right] == answer[right]):
                right += 1

            if right == len(answer):
                end = i + right - 1
                break

    return start, end


def gen_trainable_dataset(sample, debug=False):
    trainable_sample = {
        'question_id': sample['question_id'],
        'fact_or_opinion': sample['fact_or_opinion'],
        'question_type': sample['question_type'],
        'segmented_question': sample['segmented_question'],
        'pos_question': sample['pos_question'],
        'keyword_question': sample['keyword_question'],
        'documents': sample['documents']
    }

    # test data
    if 'segmented_answers' not in sample:
        return trainable_sample

    trainable_sample['segmented_answers'] = sample['segmented_answers']

    # 为每个answer生成对应的 best_match_doc、labels和 fake answer
    multi_best_match_doc_ids = []
    multi_best_start_end_idx = []
    multi_best_fake_answers = []
    multi_best_match_score = []

    for answer in sample['segmented_answers']:
        if answer == '' or len(answer) == 0: continue

        answer_tokens = set([token for token in answer])
        ques_answer = [sample['segmented_question'] + answer]

        best_match_score = 0
        best_match_doc_id = -1
        best_match_start_idx = -1
        best_match_end_idx = -1
        best_fake_answer = ''
        for doc_id, doc in enumerate(sample['documents']):
            if not doc['is_selected'] or len(doc['segmented_passage']) == 0:
                continue

            # 如果答案是passage的一个子数组，则直接定位
            sub_start_idx, sub_end_idx = contain_sublist(doc['segmented_passage'], answer)
            if sub_start_idx != -1 and sub_end_idx != -1:
                best_match_score = 1
                best_match_doc_id = doc_id
                best_match_start_idx = sub_start_idx
                best_match_end_idx = sub_end_idx
                best_fake_answer = doc['segmented_passage'][best_match_start_idx: best_match_end_idx + 1]
                break

            # 标题不检索
            from_start = doc['title_len'] + 1  # from_start = 0 标题参与检索
            for start_idx in range(from_start, len(doc['segmented_passage'])):
                if doc['segmented_passage'][start_idx] not in answer_tokens:
                    continue

                for end_idx in range(len(doc['segmented_passage']) - 1, start_idx - 1, -1):
                    if doc['segmented_passage'][end_idx] not in answer_tokens:
                        continue

                    span_tokens = doc['segmented_passage'][start_idx: end_idx + 1]
                    # 构造 MRC 数据集的时候只采用 f1，bleu计算太慢
                    match_score = metric_max_over_ground_truths(f1_score, None, span_tokens, ques_answer)

                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_doc_id = doc_id
                        best_match_start_idx = start_idx
                        best_match_end_idx = end_idx
                        best_fake_answer = span_tokens

        multi_best_match_doc_ids.append(best_match_doc_id)
        multi_best_match_score.append(best_match_score)
        multi_best_start_end_idx.append([best_match_start_idx, best_match_end_idx])
        multi_best_fake_answers.append(best_fake_answer)

    trainable_sample['best_match_doc_ids'] = multi_best_match_doc_ids
    trainable_sample['best_match_scores'] = multi_best_match_score
    trainable_sample['answer_labels'] = multi_best_start_end_idx
    trainable_sample['fake_answers'] = multi_best_fake_answers

    return trainable_sample


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        trainable_sample = gen_trainable_dataset(sample, debug=False)
        print(json.dumps(trainable_sample, ensure_ascii=False))
