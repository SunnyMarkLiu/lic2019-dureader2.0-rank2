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
        if not doc['is_selected']:
            continue

        # 从段落筛选阶段得到的 paragraph_match_score 的最大值的段落开始检索，优化检索范围
        paras = split_list_by_specific_value(doc['segmented_passage'], (u'<splitter>',))
        most_related_para_id = doc['most_related_para_id']
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

    # 根据 offset 计算真实的 start 和 end
    offset_len = sum(len(sample['documents'][doc_id]['segmented_passage']) for doc_id in range(best_match_doc_id))
    best_match_doc_idx = (best_match_start_idx + offset_len, best_match_end_idx + offset_len)

    trainable_sample['best_match_doc_id'] = best_match_doc_id
    trainable_sample['labels'] = best_match_doc_idx
    trainable_sample['fake_answer'] = best_fake_answer

    if debug:
        passage = []
        for doc in trainable_sample['documents']:
            passage += doc['segmented_passage']

        fake_answer = ''.join(passage[best_match_doc_idx[0]: best_match_doc_idx[1]+1])

        print('fake answer:')
        print(fake_answer)
        print('true label:')
        for ans in sample['segmented_answers']:
            ans = ''.join(ans)
            print(ans)
        print()

    return trainable_sample


if __name__ == '__main__':
    for line in sys.stdin:
        sample = json.loads(line.strip())
        trainable_sample = gen_trainable_dataset(sample, debug=False)
        print(json.dumps(trainable_sample, ensure_ascii=False))
