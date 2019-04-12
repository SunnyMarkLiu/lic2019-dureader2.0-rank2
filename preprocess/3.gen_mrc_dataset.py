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

    answer_tokens = set()
    for segmented_answer in sample['segmented_answers']:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])

    ques_answers = [sample['segmented_question'] + answer for answer in sample['segmented_answers'] if answer != '']

    best_match_score = 0
    best_match_doc_id = -1
    best_match_start_idx = -1
    best_match_end_idx = -1
    best_fake_answer = ''
    for doc_id, doc in enumerate(sample['documents']):
        if not doc['is_selected'] or len(doc['segmented_passage']) == 0:
            continue

        # 从段落筛选阶段得到的 paragraph_match_score 的最大值的段落开始检索，优化检索范围
        paras = split_list_by_specific_value(doc['segmented_passage'], (u'<splitter>',))
        most_related_para_id = doc['most_related_para_id']
        if doc['segmented_passage'][0] == '<splitter>':     # doc 的 title 为空，但多了个 <splitter>
            most_related_para_id -= 1
        most_related_para_tokens = paras[most_related_para_id]

        para_offset_len = sum(len(paras[para_id]) for para_id in range(most_related_para_id)) + most_related_para_id

        for start_idx in range(len(most_related_para_tokens)):
            if most_related_para_tokens[start_idx] not in answer_tokens:
                continue

            for end_idx in range(len(most_related_para_tokens) - 1, start_idx - 1, -1):
                if most_related_para_tokens[end_idx] not in answer_tokens:
                    continue

                span_tokens = most_related_para_tokens[start_idx: end_idx + 1]
                # 构造 MRC 数据集的时候只采用 f1，bleu计算太慢
                match_score = metric_max_over_ground_truths(f1_score, None, span_tokens, ques_answers)

                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_doc_id = doc_id
                    best_match_start_idx = start_idx + para_offset_len
                    best_match_end_idx = end_idx + para_offset_len
                    best_fake_answer = span_tokens

    best_start_end_idx = [best_match_start_idx, best_match_end_idx]

    trainable_sample['gold_answer'] = {
        'best_match_doc_id': best_match_doc_id,
        'best_match_score': best_match_score,
        'labels': best_start_end_idx,
        'fake_answer': best_fake_answer
    }

    if debug:
        passage = []
        for doc in trainable_sample['documents']:
            passage += doc['segmented_passage']

        fake_answer = ''.join(sample['documents'][best_match_doc_id]['segmented_passage']\
                                  [best_start_end_idx[0]: best_start_end_idx[1]+1])

        print('fake answer:')
        print(fake_answer)
        print('true label:')
        for ans in sample['segmented_answers']:
            ans = ''.join(ans)
            print(ans)
        print()

    # 为每个answer生成对应的 best_match_doc、labels和 fake answer
    multi_best_match_doc_ids = []
    multi_best_start_end_idx = []
    multi_best_fake_answers = []
    multi_best_match_score = []

    for answer in sample['segmented_answers']:
        if answer == '': continue

        ques_answers = [sample['segmented_question'] + answer]

        best_match_score = 0
        best_match_doc_id = -1
        best_match_start_idx = -1
        best_match_end_idx = -1
        best_fake_answer = ''
        for doc_id, doc in enumerate(sample['documents']):
            if not doc['is_selected'] or len(doc['segmented_passage']) == 0:
                continue

            # 从段落筛选阶段得到的 paragraph_match_score 的最大值的段落开始检索，优化检索范围
            paras = split_list_by_specific_value(doc['segmented_passage'], (u'<splitter>',))
            most_related_para_id = doc['most_related_para_id']
            if doc['segmented_passage'][0] == '<splitter>':  # doc 的 title 为空，但多了个 <splitter>
                most_related_para_id -= 1
            most_related_para_tokens = paras[most_related_para_id]

            para_offset_len = sum(len(paras[para_id]) for para_id in range(most_related_para_id)) + most_related_para_id

            for start_idx in range(len(most_related_para_tokens)):
                if most_related_para_tokens[start_idx] not in answer_tokens:
                    continue

                for end_idx in range(len(most_related_para_tokens) - 1, start_idx - 1, -1):
                    if most_related_para_tokens[end_idx] not in answer_tokens:
                        continue

                    span_tokens = most_related_para_tokens[start_idx: end_idx + 1]
                    # 构造 MRC 数据集的时候只采用 f1，bleu计算太慢
                    match_score = metric_max_over_ground_truths(f1_score, None, span_tokens, ques_answers)

                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_doc_id = doc_id
                        best_match_start_idx = start_idx + para_offset_len
                        best_match_end_idx = end_idx + para_offset_len
                        best_fake_answer = span_tokens

        multi_best_match_score.append(best_match_score)
        multi_best_match_doc_ids.append(best_match_doc_id)
        multi_best_start_end_idx.append([best_match_start_idx, best_match_end_idx])
        multi_best_fake_answers.append(best_fake_answer)

    trainable_sample['multi_answers'] = {
        'best_match_doc_id': multi_best_match_doc_ids,
        'best_match_score': multi_best_match_score,
        'labels': multi_best_start_end_idx,
        'fake_answer': multi_best_fake_answers
    }

    return trainable_sample


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        trainable_sample = gen_trainable_dataset(sample, debug=False)
        print(json.dumps(trainable_sample, ensure_ascii=False))
