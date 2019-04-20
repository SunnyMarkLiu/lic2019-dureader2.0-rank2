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
import numpy as np
from utils.metric_util import metric_max_over_ground_truths, f1_score
import warnings
from zhon.hanzi import punctuation
warnings.filterwarnings("ignore")
punc_filtered = set(punctuation)
punc_filtered.add(u'<splitter>')


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

    if 'entity_answers' in sample:
        trainable_sample['entity_answers'] = sample['entity_answers']

    if len(sample['segmented_answers']) == 0:
        return None

    trainable_sample['segmented_answers'] = sample['segmented_answers']

    # 为每个answer生成对应的 best_match_doc、labels和 fake answer
    multi_best_match_doc_ids = []
    multi_best_start_end_idx = []
    multi_best_fake_answers = []
    multi_best_match_score = []

    for answer in trainable_sample['segmented_answers']:
        if answer == '' or len(answer) == 0: continue

        answer_tokens = set([token for token in answer])
        ques_answer = [trainable_sample['segmented_question'] + answer]

        best_match_score = 0
        best_match_doc_id = -1
        best_match_start_idx = -1
        best_match_end_idx = -1
        best_fake_answer = ''
        for doc_id, doc in enumerate(trainable_sample['documents']):
            # is_selected 并不准确，此处不能将 selected 为 false 的 doc 过滤掉
            # if not doc['is_selected'] or len(doc['segmented_passage']) == 0:
            if len(doc['segmented_passage']) == 0:
                continue

            # ---------------- 直接定位答案 ----------------
            sub_start_idx, sub_end_idx = contain_sublist(doc['segmented_passage'], answer)
            if sub_start_idx == -1 or sub_end_idx == -1:
                if answer[-1] == '。':
                    # 去掉末尾的句号再查找
                    sub_start_idx, sub_end_idx = contain_sublist(doc['segmented_passage'], answer[:-1])
                    if sub_start_idx == -1 or sub_end_idx == -1:
                        if ',' in doc['segmented_passage'] or ';' in doc['segmented_passage']:
                            doc_passage = np.array(doc['segmented_passage'])
                            doc_passage[doc_passage == ','] = '，'
                            doc_passage[doc_passage == ';'] = '；'
                            if sum(doc_passage != np.array(doc['segmented_passage']).tolist()) > 0:  # 转换了再开始查找
                                sub_start_idx, sub_end_idx = contain_sublist(doc_passage, answer[:-1])
                else:
                    # 将英文的标点符号,->，;->；再次定位查找
                    if ',' in doc['segmented_passage'] or ';' in doc['segmented_passage']:
                        doc_passage = np.array(doc['segmented_passage'])
                        doc_passage[doc_passage == ','] = '，'
                        doc_passage[doc_passage == ';'] = '；'
                        if sum(doc_passage != np.array(doc['segmented_passage']).tolist()) > 0:   # 转换了再开始查找
                            sub_start_idx, sub_end_idx = contain_sublist(doc_passage, answer)

            if sub_start_idx != -1 and sub_end_idx != -1:   # 定位到了答案
                best_match_score = 1
                best_match_doc_id = doc_id
                best_match_start_idx = sub_start_idx
                best_match_end_idx = sub_end_idx
                best_fake_answer = doc['segmented_passage'][best_match_start_idx: best_match_end_idx + 1]
                break
            else:
                # 去掉<splitter>的答案直接定位，对于跨段落的答案
                clean_passage = [token for token in doc['segmented_passage'] if token != '<splitter>']
                sub_start_idx, sub_end_idx = contain_sublist(clean_passage, answer)

                if sub_start_idx == -1 or sub_end_idx == -1:
                    if answer[-1] == '。':
                        # 去掉末尾的句号再查找
                        sub_start_idx, sub_end_idx = contain_sublist(clean_passage, answer[:-1])
                        if sub_start_idx == -1 or sub_end_idx == -1:
                            if ',' in clean_passage or ';' in clean_passage:
                                doc_passage = np.array(clean_passage)
                                doc_passage[doc_passage == ','] = '，'
                                doc_passage[doc_passage == ';'] = '；'
                                if sum(doc_passage != np.array(clean_passage).tolist()) > 0:  # 转换了再开始查找
                                    sub_start_idx, sub_end_idx = contain_sublist(doc_passage, answer[:-1])
                    else:
                        # 将英文的标点符号,->，;->；再次定位查找
                        if ',' in clean_passage or ';' in clean_passage:
                            doc_passage = np.array(clean_passage)
                            doc_passage[doc_passage == ','] = '，'
                            doc_passage[doc_passage == ';'] = '；'
                            if sum(doc_passage != np.array(clean_passage).tolist()) > 0:  # 转换了再开始查找
                                sub_start_idx, sub_end_idx = contain_sublist(doc_passage, answer)

                if sub_start_idx != -1 and sub_end_idx != -1:   # 定位到了答案
                    best_match_score = 1
                    best_match_doc_id = doc_id
                    best_match_start_idx = sub_start_idx
                    best_match_end_idx = sub_end_idx
                    best_fake_answer = clean_passage[best_match_start_idx: best_match_end_idx + 1]

                    # 定位 <splitter> 的下标进行删除
                    splitter_idx = [index for index, value in enumerate(doc['segmented_passage']) if value == '<splitter>']
                    doc['segmented_passage'] = clean_passage
                    doc['pos_passage'] = [token for token in doc['pos_passage'] if token != '<splitter>']
                    doc['keyword_passage'] = [value for index, value in enumerate(doc['keyword_passage']) if index not in splitter_idx]
                    doc['passage_word_in_question'] = [value for index, value in enumerate(doc['passage_word_in_question']) if index not in splitter_idx]
                    break
            # -------------------- 直接定位答案 end ----------------------------

            # 如果第一个段落中问题长度内不包含问题（test/dev）、问题+答案（train）的关键词，则从标题检索，from_start=0
            # 如果第一个段落中问题长度内包含，则标题不检索 from_start = doc['title_len'] + 1
            check_para1_contex_start = doc['title_len'] + 1
            check_para1_contex_end = doc['title_len'] + 1 + len(ques_answer[0])

            para1_pre_context = doc['segmented_passage'][check_para1_contex_start: check_para1_contex_end]
            para1_pre_context_words = set([token for token in para1_pre_context])

            if len(set(ques_answer[0]).intersection(para1_pre_context_words)) > 0:  # TODO 只取 keywords
                from_start = doc['title_len'] + 1
            else:
                from_start = 0

            for start_idx in range(from_start, len(doc['segmented_passage'])):
                # 开始的词不在答案中，或者，开始的词为标点符号或splitter，直接过滤
                if doc['segmented_passage'][start_idx] not in answer_tokens or \
                        doc['segmented_passage'][start_idx] in punc_filtered:
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
        if trainable_sample is not None:
            print(json.dumps(trainable_sample, ensure_ascii=False))
