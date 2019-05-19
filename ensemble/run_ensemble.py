#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/15 20:32
"""
import sys
sys.path.append('../tfmrc/')
import os
import logging
import argparse
import json
from tqdm import tqdm
from util.metric import normalize, compute_bleu_rouge
from ensemble_dataset import EnsembleDataset
from collections import defaultdict
from answer_text_norm import AnswerNormer


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description='Reading Comprehension on BaiduRC dataset')

    parser.add_argument('--model_predicts', nargs='+',
                        # 模型顺序是dev分数最高的排在前面
                        default=[
                            # '../backup/tfmrc_5_08_57.81/',
                            '../backup/tfmrc_5_10_58.2/',
                            # '../backup/tfmrc_5_13_rnet_58.47/',
                            # '../backup/tfmrc_5_16_full_datas_pretrain_58.62/',
                            '../backup/tfmrc_5_14_rnet_dropout_58.76/',
                            # '../backup/tfmrc_5_14_rnet_dropout_58.76_round2/',
                            # '../backup/tfmrc_5_15_new_vocab_59.25/',
                            '../backup/tfmrc_5_15_new_vocab_59.25_round2/',
                            '../backup/tfmrc_5_16_full_datas_pretrain_58.62_round2/'
                            ],
                        help='list of files that current great models predicted')
    parser.add_argument('--mode', type=str, choices=['dev', 'test', 'test2'],
                        help='ensemble for dev or test')
    parser.add_argument('--data_type', type=str, choices=['search', 'zhidao'],
                        help='the type of the data, search or zhidao')
    parser.add_argument('--max_p_num', type=int, default=5,
                        help='max passage num in one sample')
    parser.add_argument('--max_p_len', type=int, default=500,
                        help='max length of passage')
    parser.add_argument('--max_q_len', type=int, default=20,
                        help='max length of question')
    parser.add_argument('--max_a_len', type=int, default=300,  # search：300，zhidao：400
                        help='max length of answer')

    parser.add_argument('--dev_file', type=str,
                        default='../input/dureader_2.0_v5/final_mrc_dataset/devset/{}.{}.json',
                        help='preprocessed test file')
    parser.add_argument('--test_file', type=str,
                        default='../input/dureader_2.0_v5/final_mrc_dataset/testset/{}.{}.json',
                        help='preprocessed test file')
    parser.add_argument('--test2_file', type=str,
                        default='../input/dureader_2.0_v5/final_mrc_dataset/test2set/{}.{}.json',
                        help='preprocessed test file')

    parser.add_argument('--result_dir', default='./',
                               help='the dir to output the results')

    # 文档rank分数选择
    parser.add_argument('--use_para_prior_scores', default='None',
                        choices=["None", "baidu", "zhidao", "search", "all", "best"],
                        help='document preranking score')
    return parser.parse_args()


def evaluate(args, total_batch_count, eval_batches, answer_normer, doc_prerank_mode, result_dir=None, result_prefix=None,
             save_full_info=False):
    """
    Evaluates the model performance on eval_batches and results are saved if specified
    Args:
        args: arguments
        total_batch_count: total batch counts
        eval_batches: iterable batch data
        answer_normer: predict answer normalizer
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
        for sample, start_probs, end_probs in zip(batch['raw_data'], batch['start_probs'], batch['end_probs']):

            # 针对每个模型预测的概率计算最佳的开始结束下标
            answer_phrase_score = defaultdict(float)
            for start_prob, end_prob in zip(start_probs, end_probs):
                find_best_answer_for_sample(args, sample, start_prob, end_prob, padded_p_len, pp_scores, answer_phrase_score, answer_normer)

            if len(answer_phrase_score) > 0:
                best_answer = max(answer_phrase_score.items(), key=lambda pair: pair[1])[0]
            else:
                best_answer = ''

            if save_full_info:
                sample['pred_answers'] = [best_answer]
                pred_answers.append(sample)
            else:
                pred_answers.append({'question_id': sample['question_id'],
                                     'question_type': sample['question_type'],
                                     'answers': [best_answer],
                                     'entity_answers': [[]],
                                     'yesno_answers': [],
                                     'segmented_question': sample['segmented_question']})
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


def find_best_answer_for_sample(args, sample, start_prob, end_prob, padded_p_len, para_prior_scores, answer_phrase_score, answer_normer):
    """
    Finds the best answer for a sample given start_prob and end_prob for each position.
    This will call find_best_answer_for_passage because there are multiple passages in a sample
    """
    for p_idx, passage in enumerate(sample['documents']):
        if p_idx >= args.max_p_num:
            continue
        passage_len = min(args.max_p_len, len(passage['segmented_passage']))
        find_best_answer_for_passage(args,
                                     start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                                     end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                                     para_prior_scores,
                                     p_idx,
                                     answer_phrase_score,
                                     passage['segmented_passage'],
                                     answer_normer,
                                     passage_len)

def find_best_answer_for_passage(args, start_probs, end_probs, para_prior_scores, p_idx, answer_phrase_score,
                                 seg_passage, answer_normer, passage_len=None):
    """
    Finds the best answer with the maximum start_prob * end_prob from a single passage
    """
    if passage_len is None:
        passage_len = len(start_probs)
    else:
        passage_len = min(len(start_probs), passage_len)
    best_start, best_end, max_prob = -1, -1, 0
    for start_idx in range(passage_len):
        for ans_len in range(args.max_a_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_probs[start_idx] * end_probs[end_idx]
            if prob > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob

    if para_prior_scores is not None:
        max_prob *= para_prior_scores[p_idx]

    # norm 处理
    if best_start >= 0 and best_end < len(seg_passage):
        predict_answer = answer_normer.norm_predict_answer(seg_passage[best_start: best_end + 1])
        answer_phrase_score[predict_answer] += max_prob

def ensemble():
    """
    ensemble all model predicted results with specific weights
    """
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info('Running with args: ')
    logger.info('*' * 100)
    for k, v in vars(args).items():
        logger.info('   {}:\t{}'.format(k, v))
    logger.info('*' * 100)

    # 待预测文件
    if args.mode == 'dev':
        test_file = args.dev_file.format(args.data_type, args.mode)
    elif args.mode == 'test':
        test_file = args.test_file.format(args.data_type, args.mode + '1')
    elif args.mode == 'test2':
        test_file = args.test2_file.format(args.data_type, args.mode)
    else:
        raise ValueError('Inconsistency between mode and test file')
    logger.info('test file: {}'.format(test_file))

    # 所有模型预测的结果文件
    predict_json = args.mode + '.predicted.json'
    predict_test_files = [os.path.join(predict, 'cache/results/', args.data_type, predict_json) for predict in args.model_predicts]
    logger.info('ensemble model predicted results: {}'.format(predict_test_files))

    ensemble_data = EnsembleDataset(max_p_num=args.max_p_num,
                                    max_p_len=args.max_p_len,
                                    max_q_len=args.max_q_len,
                                    max_a_len=args.max_a_len,
                                    test_file=test_file,
                                    predicted_test_files=predict_test_files)
    batch_generator = ensemble_data.gen_test_mini_batches(batch_size=128)
    total_batch_count = ensemble_data.get_data_length() // 128 + int(ensemble_data.get_data_length() % 128 != 0)

    # 预测答案进行标准化
    if args.mode == 'dev' or args.mode == 'test':
        url_map_path = '../input/dureader_2.0_v5/url_mapping.csv'
    elif args.mode == 'test2':
        url_map_path = '../input/dureader_2.0_v5/url_mapping_test2.csv'
    else:
        raise ValueError('url id mapping file error!')
    logger.info('using url id mapping file: {}'.format(url_map_path))
    answer_normer = AnswerNormer(url_map_path=url_map_path)

    # 结果保存文件前缀
    result_prefix = '{}.{}.ensemble'.format(args.data_type, args.mode)
    logger.info('Predicted answers will be saved to {}'.format(os.path.join(args.result_dir, result_prefix)))

    bleu_rouge = evaluate(args, total_batch_count, batch_generator, answer_normer, doc_prerank_mode=args.use_para_prior_scores,
                          result_dir=args.result_dir, result_prefix=result_prefix)
    if bleu_rouge:
        logger.info('Result on dev set: {}'.format(bleu_rouge))
    logger.info('done.')


if __name__ == '__main__':
    ensemble()
