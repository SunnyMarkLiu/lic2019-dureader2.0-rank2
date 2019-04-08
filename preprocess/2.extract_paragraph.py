#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
针对每个文档进行段落筛选。如果段落过长，则划分为多个句子，进行句子筛选。

对于训练集：问题 + 答案，进行段落筛选
对于验证集和测试集：只针对问题解析段落筛选

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/5 14:22
"""
import sys
import json
from collections import Counter


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truths: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def calc_paragraph_match_scores(doc, question, answers=None):
    """
    Train mode: For each document, calculate the match score between paragraph and question with answers.
    Test/Dev Mode: For each document, calculate the match score between paragraph and question.
    """
    match_scores = []
    if answers:
        ques_answers = [question + answer for answer in answers]
    else:
        ques_answers = [question]

    for para_id, para_tokens in enumerate(doc['segmented_paragraphs']):
        # 问题 + 答案组成的查询语句，baseline 中只采用答案，对于答案较短的情况存在缺陷
        related_score = metric_max_over_ground_truths(recall, para_tokens, ques_answers)
        match_scores.append(related_score)
    return match_scores


def extract_paragraph(sample, mode, max_doc_len, match_score_threshold):
    """
    对于训练集，计算每个 doc 的每个段落 para 与 question+answer 的 f1 值
    对于测试集和验证集，计算每个 doc 的每个段落 para 与 question 的 f1 值
    Args:
        sample: a sample in the dataset.
        mode: string of ("train", "dev", "test"), indicate the type of dataset to process.
    """
    question = sample['segmented_question']
    if mode == 'train':
        answers = sample['segmented_answers']
    else:  # dev/test
        answers = None

    for doc_id, doc in enumerate(sample['documents']):
        # 计算每个doc的 paragraph 和查询（question、question+answer）的 f1 值
        para_match_scores = calc_paragraph_match_scores(doc, question, answers)
        para_infos = []
        for p_idx, (para_tokens, para_score) in enumerate(zip(doc['segmented_paragraphs'], para_match_scores)):
            # ((段落匹配得分，段落长度)，段落的原始下标)
            para_infos.append((para_score, len(para_tokens), p_idx))

        last_para_id = -1
        last_para_cut_idx = -1
        selected_para_ids = []

        # 按照 match_score 降序排列，按照段落长度升序排列
        para_infos.sort(key=lambda x: (-x[0], x[1]))

        selected_para_len = len(doc['segmented_title'])  # 注意拼接上 title
        for para_info in para_infos:
            # 过滤掉匹配得分小于阈值的段落
            if para_info[0] < match_score_threshold:
                continue

            para_id = para_info[-1]
            selected_para_len += len(doc['segmented_paragraphs'][para_id])
            if selected_para_len <= max_doc_len:
                selected_para_ids.append(para_id)
            else:
                # 对于超出最大 doc 长度的，截取到最大长度，baseline选取 top3，可能筛掉了答案所在的段落
                last_para_id = para_id
                last_para_cut_idx = max_doc_len - selected_para_len
                break

        # para 原始顺序
        selected_para_ids.sort()

        segmented_paragraphs = [doc['segmented_paragraphs'][i] for i in selected_para_ids]
        paragraph_match_scores = [para_match_scores[i] for i in selected_para_ids]
        pos_paragraphs = [doc['pos_paragraphs'][i] for i in selected_para_ids]
        keyword_paragraphs = [doc['keyword_paragraphs'][i] for i in selected_para_ids]

        if last_para_id > -1:
            last_seg_para = doc['segmented_paragraphs'][last_para_id][:last_para_cut_idx]
            segmented_paragraphs.append(last_seg_para)

            if answers:  # train mode
                ques_answers = [question + answer for answer in answers]
            else:   # dev or test mode
                ques_answers = [question]
            paragraph_match_scores.append(metric_max_over_ground_truths(recall, last_seg_para, ques_answers))
            pos_paragraphs.append(doc['pos_paragraphs'][last_para_id][:last_para_cut_idx])
            keyword_paragraphs.append(doc['keyword_paragraphs'][last_para_id][:last_para_cut_idx])

        # update
        doc['segmented_paragraphs'] = segmented_paragraphs
        doc['paragraph_match_score'] = paragraph_match_scores
        doc['pos_paragraphs'] = pos_paragraphs
        doc['keyword_paragraphs'] = keyword_paragraphs

if __name__ == '__main__':
    # mode="train"/"dev"/"test"
    mode = sys.argv[1]
    max_doc_len = int(sys.argv[2])
    match_score_threshold = float(sys.argv[3])

    for line in sys.stdin:
        sample = json.loads(line.strip())
        extract_paragraph(sample, mode, max_doc_len, match_score_threshold)
        print(json.dumps(sample, ensure_ascii=False))
