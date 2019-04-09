#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/9 11:51
"""
import nltk
from collections import Counter


def _precision_recall_f1(prediction, ground_truth):
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
    return _precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    """
    return _precision_recall_f1(prediction, ground_truth)[2]


def bleu_4(prediction, ground_truths):
    """
    This function calculates and returns the bleu-4
    """
    if len(prediction) == 0 and len(ground_truths) != 0 and sum([len(gt) for gt in ground_truths]) != 0:
        return 0

    if len(prediction) < 4:
        weights = [1 / len(prediction)] * len(prediction)
        bleu_score = nltk.translate.bleu_score.sentence_bleu(ground_truths, prediction, weights=weights)
    else:
        bleu_score = nltk.translate.bleu_score.sentence_bleu(ground_truths, prediction)
    return bleu_score


def metric_max_over_ground_truths(f1_score_fn, bleu4_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    """
    f1_for_ground_truths = []
    for ground_truth in ground_truths:
        f1 = f1_score_fn(prediction, ground_truth)
        f1_for_ground_truths.append(f1)

    max_f1 = max(f1_for_ground_truths)
    if bleu4_fn is not None:
        bleu = bleu4_fn(prediction, ground_truths)
    else:
        bleu = 1
    return max_f1 * bleu


def metric_over_ground_truth(metric_fn, prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    """
    return metric_fn(prediction, ground_truth)
