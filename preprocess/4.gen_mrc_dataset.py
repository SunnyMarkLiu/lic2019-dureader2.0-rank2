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
from utils.metric_util import read_data_to_dict, compute_bleu_rouge
from zhon.hanzi import punctuation
from check.metric.rouge import RougeL
from check.metric.bleu import BLEUWithBonus
import warnings
warnings.filterwarnings("ignore")

punc_filtered = set(punctuation)
punc_filtered.add(u'<splitter>')

alpha, beta = 1, 1
bleu_eval = BLEUWithBonus(4, alpha=alpha, beta=beta)
rouge_eval = RougeL(alpha=alpha, beta=beta, gamma=1.2)

def calc_one_sample_metric(sample):
    """ 计算一个样本的 rouge-l 和 bleu4 分数 """
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


def gen_trainable_dataset(sample):
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

    if len(sample['segmented_answers']) == 0 or len(sample['documents']) == 0:
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
        best_fake_answer = ['']
        for doc_id, doc in enumerate(trainable_sample['documents']):
            # is_selected 并不准确，此处不能将 selected 为 false 的 doc 过滤掉
            # if not doc['is_selected'] or len(doc['segmented_passage']) == 0:
            if len(doc['segmented_passage']) == 0:
                continue

            # ---------------- 直接定位答案 start ----------------
            sub_start_idx, sub_end_idx = contain_sublist(doc['segmented_passage'], answer)
            if sub_start_idx == -1 or sub_end_idx == -1:
                if answer[-1] in {'.', '!', '?', ';', ',', ' '}:
                    # 去掉末尾的标点符号再查找
                    sub_start_idx, sub_end_idx = contain_sublist(doc['segmented_passage'], answer[:-1])

            if sub_start_idx != -1 and sub_end_idx != -1:   # 定位到了答案
                best_match_score = 1
                best_match_doc_id = doc_id
                best_match_start_idx = sub_start_idx
                best_match_end_idx = sub_end_idx
                best_fake_answer = doc['segmented_passage'][best_match_start_idx: best_match_end_idx + 1]
                break

            # 去掉<splitter>的答案直接定位，对于跨段落的答案
            clean_passage = [token for token in doc['segmented_passage'] if token != '<splitter>']
            sub_start_idx, sub_end_idx = contain_sublist(clean_passage, answer)

            if sub_start_idx == -1 or sub_end_idx == -1:
                if answer[-1] in {'.', '!', '?', ';', ',', ' '}:
                    # 去掉末尾的标点符号再查找
                    sub_start_idx, sub_end_idx = contain_sublist(clean_passage, answer[:-1])

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

            if len(set(ques_answer[0]).intersection(para1_pre_context_words)) > 0:
                from_start = doc['title_len'] + 1
            else:
                from_start = 0

            for start_idx in range(from_start, len(doc['segmented_passage'])):
                # 开始的词不在答案中，或者，开始的词为标点符号或splitter，直接过滤
                if doc['segmented_passage'][start_idx] not in answer_tokens or \
                        (doc['segmented_passage'][start_idx] in punc_filtered and doc['segmented_passage'][start_idx] not in {'<', '(', '《', '"'}):
                    continue

                for end_idx in range(len(doc['segmented_passage']) - 1, start_idx - 1, -1):
                    # 结尾的 end_idx 后的连续三个词都不在答案中，扩大答案搜索的范围，防止因为某个词 fake answer 在 answer 中间断掉
                    if (doc['segmented_passage'][end_idx] not in answer_tokens) and \
                       (end_idx + 1 < len(doc['segmented_passage']) and doc['segmented_passage'][end_idx + 1] not in answer_tokens) and \
                       (end_idx + 2 < len(doc['segmented_passage']) and doc['segmented_passage'][end_idx + 2] not in answer_tokens):
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

        # best_match_doc_id 的 is_selected 设置为 true，如果为false，可实现进行纠正
        if -1 < best_match_doc_id < len(trainable_sample['documents']):
            trainable_sample['documents'][best_match_doc_id]['is_selected'] = True

    trainable_sample['best_match_doc_ids'] = multi_best_match_doc_ids
    trainable_sample['best_match_scores'] = multi_best_match_score
    trainable_sample['answer_labels'] = multi_best_start_end_idx
    trainable_sample['fake_answers'] = multi_best_fake_answers

    # 计算当前抽取到的 fake asnwer 的 ROUGE-L 和 BLEU4 分数
    rouge_l, bleu4 = calc_one_sample_metric(trainable_sample)
    trainable_sample['ceil_rouge_l'] = rouge_l
    trainable_sample['ceil_bleu4'] = bleu4

    return trainable_sample


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        try:
            sample = json.loads(line.strip())
        except:
            continue

        sample = gen_trainable_dataset(sample)
        if sample is not None:
            print(json.dumps(sample, ensure_ascii=False))
