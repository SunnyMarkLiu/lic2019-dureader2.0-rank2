#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/15 20:32
"""
import os
import logging
import argparse
import json
from tqdm import tqdm
from util.metric import normalize
from util.metric import compute_bleu_rouge
from ensemble.ensemble_dataset import EnsembleDataset


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Reading Comprehension on BaiduRC dataset')

    parser.add_argument('--model_predicts', nargs='+',
                        default=['../backup/tfmrc_5_12_56.93/',
                                 '../backup/tfmrc_5_08_57.81/',
                                 '../backup/tfmrc_5_10_58.2/',
                                 '../backup/tfmrc_5_13_rnet_58.47/',
                                 '../backup/tfmrc_5_14_rnet_dropout_58.76/'],
                        help='list of files that current great models predicted')
    parser.add_argument('--model_weights', nargs='+',
                        default=[0.2, 0.2, 0.2, 0.2, 0.2],
                        help='the average weights of models')
    parser.add_argument('--mode', type=str, choices=['dev', 'test'],
                        help='ensemble for dev or test')
    parser.add_argument('--data_type', type=str,
                        help='the type of the data, search or zhidao')
    parser.add_argument('--max_p_num', type=int, default=5,
                        help='max passage num in one sample')
    parser.add_argument('--max_p_len', type=int, default=500,
                        help='max length of passage')
    parser.add_argument('--max_q_len', type=int, default=20,
                        help='max length of question')
    parser.add_argument('--max_a_len', type=int, default=300,  # search：300，zhidao：400
                        help='max length of answer')
    # 文档rank分数选择
    parser.add_argument('--use_para_prior_scores', default='None',
                        choices=["None", "baidu", "zhidao", "search", "all", "best"],
                        help='document preranking score')
    return parser.parse_args()


def evaluate(total_batch_count, eval_batches, doc_prerank_mode, result_dir=None, result_prefix=None,
             save_full_info=False):
    """
    Evaluates the model performance on eval_batches and results are saved if specified
    Args:
        total_batch_count: total batch counts
        eval_batches: iterable batch data
        doc_prerank_mode: document prerank scores
        result_dir: directory to save predicted answers, answers will not be saved if None
        result_prefix: prefix of the file for saving predicted answers,
                       answers will not be saved if None
        save_full_info: if True, the pred_answers will be added to raw sample and saved
    """
    logger = logging.getLogger()

    pred_answers, ref_answers = [], []

    if doc_prerank_mode == 'baidu':
        pp_scores = (0.44, 0.23, 0.15, 0.09, 0.07)
    elif doc_prerank_mode == 'zhidao':
        pp_scores = (
        0.41365245374094933, 0.22551086082059532, 0.15545454545454546, 0.11234915526950925, 0.09303298471440065)
    elif doc_prerank_mode == 'search':
        pp_scores = (
        0.4494739641331627, 0.2413532335000914, 0.15594927451925475, 0.09319061944255157, 0.060032908404939585)
    elif doc_prerank_mode == 'all':
        pp_scores = (0.43, 0.23, 0.16, 0.10, 0.09)
    elif doc_prerank_mode == 'best':
        pp_scores = (0.9, 0.05, 0.01, 0.0001, 0.0001)
    else:
        pp_scores = None
    logger.info('we use {} model: {} pp_scores'.format(doc_prerank_mode, pp_scores))

    tqdm_batch_iterator = tqdm(eval_batches, total=total_batch_count)
    for b_itx, batch in enumerate(tqdm_batch_iterator):
        padded_p_len = batch['pad_p_len']
        start_probs, end_probs = batch['start_probs'], batch['end_probs']
        for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

            best_answer, segmented_pred = find_best_answer(sample, start_prob, end_prob, padded_p_len, pp_scores)
            if save_full_info:
                sample['pred_answers'] = [best_answer]
                sample['start_prob'] = start_prob.tolist()
                sample['end_prob'] = end_prob.tolist()
                pred_answers.append(sample)
            else:
                pred_answers.append({'question_id': sample['question_id'],
                                     'question_type': sample['question_type'],
                                     'answers': [best_answer],
                                     'entity_answers': [[]],
                                     'yesno_answers': [],
                                     'segmented_question': sample['segmented_question'],
                                     'segmented_answers': segmented_pred,

                                     'start_prob': start_prob.tolist(),  # 保存 start 和 end 的概率，用于后期的 ensemble
                                     'end_prob': end_prob.tolist()})
            if 'segmented_answers' in sample:
                ref_answers.append({'question_id': sample['question_id'],
                                    'question_type': sample['question_type'],
                                    'answers': [''.join(seg_ans) for seg_ans in sample['segmented_answers']],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})

    if result_dir is not None and result_prefix is not None:
        result_file = os.path.join(result_dir, result_prefix + '.json')
        with open(result_file, 'w') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

        logger.info('Saving {} results to {}'.format(result_prefix, result_file))

    # compute the bleu and rouge scores if reference answers is provided
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None
    return bleu_rouge


def find_best_answer(self, sample, start_prob, end_prob, padded_p_len, para_prior_scores=None):
    """
    Finds the best answer for a sample given start_prob and end_prob for each position.
    This will call find_best_answer_for_passage because there are multiple passages in a sample
    """
    best_p_idx, best_span, best_score = None, None, 0
    for p_idx, passage in enumerate(sample['documents']):
        if p_idx >= self.max_p_num:
            continue
        passage_len = min(self.max_p_len, len(passage['segmented_passage']))
        answer_span, score = self.find_best_answer_for_passage(
            start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
            end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
            passage_len)
        if para_prior_scores is not None:
            # the Nth prior score = the Number of training samples whose gold answer comes
            #  from the Nth paragraph / the number of the training samples
            score *= para_prior_scores[p_idx]
        if score > best_score:
            best_score = score
            best_p_idx = p_idx
            best_span = answer_span
    if best_p_idx is None or best_span is None:
        best_answer = ''
        segmented_pred = []
    else:
        segmented_pred = sample['documents'][best_p_idx]['segmented_passage'][best_span[0]: best_span[1] + 1]
        best_answer = ''.join(segmented_pred)
    return best_answer, segmented_pred


def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
    """
    Finds the best answer with the maximum start_prob * end_prob from a single passage
    """
    if passage_len is None:
        passage_len = len(start_probs)
    else:
        passage_len = min(len(start_probs), passage_len)
    best_start, best_end, max_prob = -1, -1, 0
    for start_idx in range(passage_len):
        for ans_len in range(self.max_a_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_probs[start_idx] * end_probs[end_idx]
            if prob > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob
    return (best_start, best_end), max_prob


def ensemble():
    """
    ensemble all model predicted results with specific weights
    """
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args: ')
    logger.info('*' * 100)
    for k, v in vars(args).items():
        logger.info('   {}:\t{}'.format(k, v))
    logger.info('*' * 100)

    predict_json = args.mpde + '.predicted.json'
    predict_test_files = [os.path.join(predict, 'cache/results/', args.data_type, predict_json)
                          for predict in args.model_predicts]

    ensemble_data = EnsembleDataset(max_p_num=args.max_p_num,
                                    max_p_len=args.max_p_len,
                                    max_q_len=args.max_q_len,
                                    max_a_len=args.max_a_len,
                                    test_file=args.test_file,
                                    predict_test_files=predict_test_files,
                                    predict_test_weights=args.model_weights)
    batch_generator = ensemble_data.gen_test_mini_batches(batch_size=128)
    total_batch_count = ensemble_data.get_data_length() // args.batch_size + \
                        int(ensemble_data.get_data_length() % args.batch_size != 0)
    bleu_rouge = evaluate(total_batch_count, batch_generator, doc_prerank_mode=args.use_para_prior_scores,
                          result_dir=args.result_dir, result_prefix=args.result_prefix)
    if bleu_rouge:
        logger.info('Result on dev set: {}'.format(bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


if __name__ == '__main__':
    ensemble()
