#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/9 11:51
"""
import itertools
import json
import sys
import zipfile
sys.path.append('../')
from check.metric.bleu import BLEUWithBonus
from check.metric.rouge import RougeL
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from collections import Counter

bleu_smoothing_function = SmoothingFunction().method1

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

    bleu_score = sentence_bleu(ground_truths, prediction, smoothing_function=bleu_smoothing_function)
    return bleu_score


def metric_max_over_ground_truths(f1_score_fn, bleu4_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    """
    f1_for_ground_truths = []
    for ground_truth in ground_truths:
        f1 = f1_score_fn(prediction, ground_truth)
        f1_for_ground_truths.append(f1)

    # 为了综合考虑多个answer，此处将 max 改为 mean
    # max_f1 = sum(f1_for_ground_truths) / len(f1_for_ground_truths)
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

EMPTY = ''
YESNO_LABELS = {'Yes', 'No', 'Depends'}


def normalize(s):
    """
    Normalize strings to space joined chars.
    Args:
        s: a list of strings.
    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        norm_s = ''.join(tokens)
        norm_s = norm_s.replace(u"，", u",")
        norm_s = norm_s.replace(u"。", u".")
        norm_s = norm_s.replace(u"！", u"!")
        norm_s = norm_s.replace(u"？", u"?")
        norm_s = norm_s.replace(u"；", u";")
        norm_s = norm_s.replace(u"（", u"(").replace(u"）", u")")
        norm_s = norm_s.replace(u"【", u"[").replace(u"】", u"]")
        norm_s = norm_s.replace(u"“", u"\"").replace(u"“", u"\"")
        normalized.append(norm_s)
    return normalized


def data_check(obj):
    """
    Check data.

    Raises:
        Raises AssertionError when data is not legal.
    """
    assert 'question_id' in obj, "Missing 'question_id' field."
    # assert 'question_type' in obj, \
    #        "Missing 'question_type' field. question_id: {}".format(obj['question_type'])

    # assert 'yesno_answers' in obj, \
    #        "Missing 'yesno_answers' field. question_id: {}".format(obj['question_id'])
    if "yesno_answers" in obj:
        assert isinstance(obj['yesno_answers'], list), \
            r"""'yesno_answers' field must be a list, if the 'question_type' is not
            'YES_NO', then this field should be an empty list.
            question_id: {}""".format(obj['question_id'])
    else:
        obj["yesno_answers"] = []

    if "entity_answers" not in obj:
        obj["entity_answers"] = []


def read_file(file_name, is_ref=False):
    """
    Read predict answers or reference answers from file.

    Args:
        file_name: the name of the file containing predict result or reference
                   result.

    Returns:
        A dictionary mapping question_id to the result information. The result
        information itself is also a dictionary with has four keys:
        - question_type: type of the query.
        - yesno_answers: A list of yesno answers corresponding to 'answers'.
        - answers: A list of predicted answers.
        - entity_answers: A list, each element is also a list containing the entities
                    tagged out from the corresponding answer string.
    """
    def _open(file_name, mode, zip_obj=None):
        if zip_obj is not None:
            return zip_obj.open(file_name, mode)
        return open(file_name, mode)

    results = {}
    if is_ref:
        # keys = ['source', 'answers', 'yesno_answers',
        #         'entity_answers', 'question_type']
        keys = ['question_id', 'question_type', 'yesno_answers',
                'entity_answers', 'documents', 'answers']
    else:
        keys = ['answers', 'yesno_answers']
    try:
        zf = zipfile.ZipFile(
            file_name, 'r') if file_name.endswith('.zip') else None
    except:
        zf = None
    file_list = [file_name] if zf is None else zf.namelist()

    for fn in file_list:
        line_num = 0
        for line in _open(fn, 'r', zip_obj=zf):
            try:
                line_num += 1
                obj = json.loads(line.strip())
            except ValueError:
                #raise ValueError("Every line of data should be legal json, in line %s" % str(line_num))
                print >> sys.stderr, ValueError(
                    "Every line of data should be legal json, in line %s" % str(line_num))
                continue
            data_check(obj)
            qid = obj['question_id']
            assert qid not in results, "Duplicate question_id: {}".format(qid)
            results[qid] = {}
            for k in keys:
                if k == 'answers':
                    # if 'answers' not in obj:
                    #     obj[k] = [''.join(seg_ans) for seg_ans in obj['segmented_answers']]
                    results[qid][k] = normalize(obj[k])
                elif k == 'documents':
                    if 'documents' in obj:
                        results[qid][k] = obj[k]
                else:
                    results[qid][k] = obj[k]
            if is_ref:
                for i, e in enumerate(results[qid]['entity_answers']):
                    results[qid]['entity_answers'][i] = normalize(e)
    return results

def read_data_to_dict(data_list, is_ref=False):
    """
    Read predict answers or reference answers from data list to dict.
    """
    if not isinstance(data_list, list):
        # 处理只传一个sample的情况
        data_list = [data_list]
    results = {}
    if is_ref:
        # keys = ['question_id', 'answers', 'yesno_answers',
        #         'entity_answers', 'question_type']
        keys = ['question_id', 'question_type', 'yesno_answers',
                'entity_answers', 'segmented_question', 'documents', 'answers']
    else:
        keys = ['answers', 'yesno_answers']
    line_num = 0
    for sample in data_list:
        data_check(sample)
        qid = sample['question_id']
        assert qid not in results, "Duplicate question_id: {}".format(qid)
        results[qid] = {}
        for k in keys:
            if k == 'answers':
                results[qid][k] = normalize(sample[k])
            else:
                results[qid][k] = sample[k]
        if is_ref:
            for i, e in enumerate(results[qid]['entity_answers']):
                results[qid]['entity_answers'][i] = normalize(e)
    return results

def compute_bleu_rouge(pred_dict, ref_dict):
    err = None
    metrics = {}
    bleu4, rouge_l = 0.0, 0.0
    alpha = 1.0
    beta = 1.0
    bleu_eval = BLEUWithBonus(4, alpha=alpha, beta=beta)
    rouge_eval = RougeL(alpha=alpha, beta=beta, gamma=1.2)
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    try:
        for qid, results in ref_dict.items():
            cand_result = pred_dict.get(qid, {})
            #pred_answers = cand_result.get('answers', [EMPTY])[0]
            pred_answers = cand_result.get('answers', [])
            if not pred_answers:
                pred_answers = EMPTY
            else:
                pred_answers = pred_answers[0]
            pred_yn_label = None
            ref_entities = None
            ref_answers = results.get('answers', [])
            if not ref_answers:
                continue
            if results['question_type'] == 'ENTITY':
                ref_entities = set(
                    itertools.chain(*results.get('entity_answers', [[]])))
                if not ref_entities:
                    ref_entities = None
            if results['question_type'] == 'YES_NO':
                cand_yesno = cand_result.get('yesno_answers', [])
                pred_yn_label = None if len(cand_yesno) == 0 \
                    else cand_yesno[0]
            bleu_eval.add_inst(
                pred_answers,
                ref_answers,
                yn_label=pred_yn_label,
                yn_ref=results['yesno_answers'],
                entity_ref=ref_entities)
            rouge_eval.add_inst(
                pred_answers,
                ref_answers,
                yn_label=pred_yn_label,
                yn_ref=results['yesno_answers'],
                entity_ref=ref_entities)
        bleu4 = bleu_eval.score()[-1]
        rouge_l = rouge_eval.score()
    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae
    # too keep compatible to leaderboard evaluation.
    metrics['errorMsg'] = 'success' if err is None else err
    metrics['errorCode'] = 0 if err is None else 1
    metrics['ROUGE-L'] = round(rouge_l * 100, 2)
    metrics['BLEU-4'] = round(bleu4 * 100, 2)
    return metrics

def compute_bleu_rouge_onebyone(pred_dict, ref_dict, save_path):
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    score_sample_list = []
    global_bleu4, global_rouge_l = 0.0, 0.0
    skip_cnt = 0
    for qid, ref_result in ref_dict.items():
        if len(ref_result['answers']) == 0:
            skip_cnt += 1
            continue
        pred_result = pred_dict.get(qid, {})
        pred, ref = {}, {}
        pred[qid], ref[qid] = pred_result, ref_result
        metrics = compute_bleu_rouge(pred, ref)
        rouge_l, bleu4 = metrics['ROUGE-L'], metrics['BLEU-4']
        global_rouge_l += rouge_l
        global_bleu4 += bleu4
        # 存入文件的字段
        ref_result['pred_answers'] = pred_result['answers']
        ref_result['rouge_l'] = rouge_l
        ref_result['bleu4'] = bleu4
        score_sample_list.append(ref_result)
    sorted_samples = sorted(score_sample_list, key=lambda x: x['rouge_l'])
    with open(save_path, 'w') as fout:
        fout.write('\n'.join([json.dumps(sample, ensure_ascii=False) for sample in sorted_samples]))
        fout.write('\n')
    mean_rouge_l = round(global_rouge_l / (len(pred_dict) - skip_cnt), 2) if len(pred_dict) > skip_cnt else 0
    return mean_rouge_l

def main(args):
    err = None
    metrics = {}
    bleu4, rouge_l = 0.0, 0.0
    alpha = args.ab
    beta = args.ab
    bleu_eval = BLEUWithBonus(4, alpha=alpha, beta=beta)
    rouge_eval = RougeL(alpha=alpha, beta=beta, gamma=1.2)
    try:
        pred_result = read_file(args.pred_file)
        ref_result = read_file(args.ref_file, is_ref=True)
        for qid, results in ref_result.items():
            cand_result = pred_result.get(qid, {})
            #pred_answers = cand_result.get('answers', [EMPTY])[0]
            pred_answers = cand_result.get('answers', [])
            if not pred_answers:
                pred_answers = EMPTY
            else:
                pred_answers = pred_answers[0]
            pred_yn_label = None
            ref_entities = None
            ref_answers = results.get('answers', [])
            if not ref_answers:
                continue
            if results['question_type'] == 'ENTITY':
                ref_entities = set(
                    itertools.chain(*results.get('entity_answers', [[]])))
                if not ref_entities:
                    ref_entities = None
            if results['question_type'] == 'YES_NO':
                cand_yesno = cand_result.get('yesno_answers', [])
                pred_yn_label = None if len(cand_yesno) == 0 \
                    else cand_yesno[0]
            bleu_eval.add_inst(
                pred_answers,
                ref_answers,
                yn_label=pred_yn_label,
                yn_ref=results['yesno_answers'],
                entity_ref=ref_entities)
            rouge_eval.add_inst(
                pred_answers,
                ref_answers,
                yn_label=pred_yn_label,
                yn_ref=results['yesno_answers'],
                entity_ref=ref_entities)
        bleu4 = bleu_eval.score()[-1]
        rouge_l = rouge_eval.score()
    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae
    # too keep compatible to leaderboard evaluation.
    metrics['errorMsg'] = 'success' if err is None else err
    metrics['errorCode'] = 0 if err is None else 1
    metrics['data'] = [
        {'type': 'BOTH', 'name': 'ROUGE-L', 'value': round(rouge_l * 100, 2)},
        {'type': 'BOTH', 'name': 'BLEU-4', 'value': round(bleu4 * 100, 2)},
    ]
    print(json.dumps(metrics, ensure_ascii=False).encode('utf8'))
