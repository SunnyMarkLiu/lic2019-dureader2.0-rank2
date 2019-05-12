"""
This module implements data process strategies.
"""
import io
import json
import logging
import numpy as np
import pandas as pd
import itertools
from util.fine_classify import FineClassify
from random import randint


class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    id 转换和 dynamic pooling提前做好存储在数组类型的 dataset 中
    """

    def __init__(self,
                 max_p_num,
                 max_p_len,
                 max_q_len,
                 max_a_len=None,
                 train_answer_len_cut_bins=-1,
                 train_files=[],
                 dev_files=[],
                 test_files=[],
                 badcase_sample_log_file=None):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.train_answer_len_cut_bins = train_answer_len_cut_bins      # 将训练集答案平均长度切分成bin的数目，每个bin的样本大致一致

        self.badcase_sample_log_file = badcase_sample_log_file

        self.pos_meta_dict = {'nrt': 0, 'eng': 1, 'n': 2, 'f': 3, 'yg': 4, 'nt': 5, 'rr': 6, 'ad': 7, 'nr': 8, 'dg': 9,
                              't': 10, 'bg': 11, 'ag': 12, '<splitter>': 13, 'ns': 14, 'an': 15, 'b': 16, 'm': 17,
                              'v': 18, 'x': 19, 'q': 20, 'tg': 21, 'nz': 22, 'mq': 23, 'nrfg': 24, 'a': 25, 'i': 26,
                              'mg': 27, 's': 28, 'other': 29}

        # data V5 final data 频率
        self.pos_freq_dict = {'x': 0.20472713665688538, 'n': 0.19475026640706672, 'v': 0.17007803078961822,
                              'm': 0.05308703449997446, 'd': 0.045176462797625126, 'uj': 0.04279624597790906,
                              'r': 0.03424411043146353, 'c': 0.03209976815856658, 'p': 0.02723340383860834,
                              'a': 0.025835356651404726, 'eng': 0.022782153477046364, '<splitter>': 0.01980749611883006,
                              'f': 0.016547389729224673, 'vn': 0.016166487674232583, 'nr': 0.014764517903322314,
                              'ns': 0.010414494608579546, 'l': 0.007729128670414311, 't': 0.007467560328631343,
                              'ul': 0.007244628564673658, 'nz': 0.0054338941908602945, 'b': 0.005085160809003845,
                              'q': 0.004334186516791101, 'u': 0.004089915897469213, 'i': 0.003287585395998406,
                              'y': 0.0031625961887695535, 'zg': 0.002981463476789983, 'ad': 0.0025118831360989094,
                              's': 0.0024368798488865946, 'j': 0.0021248409300779997, 'ng': 0.0020861450776544833,
                              'nrt': 0.0012491703168865, 'nt': 0.001118738306843347, 'z': 0.0008952521510551211,
                              'ug': 0.0007748340613724239, 'df': 0.0007051445675547244, 'an': 0.0006737569244193399,
                              'vg': 0.0006204237329731232, 'k': 0.0006187500972582838, 'uz': 0.00048062633640901436,
                              'ud': 0.0004715119952452846, 'uv': 0.0003924152740137585, 'g': 0.00028745042076474533,
                              'mq': 0.00019329795158180943, 'e': 0.00016361183808984498, 'ag': 0.00015446611625646197,
                              'o': 0.00013763213035803535, 'nrfg': 0.0001253692620057639, 'tg': 0.0001243546203536425,
                              'h': 0.00011808545990513979, 'vd': 8.152000627697074e-05, 'yg': 5.9302492162477265e-05,
                              'rz': 1.7412784916475273e-05, 'rr': 1.4891871120998378e-05, 'vq': 1.130052781623876e-05,
                              'dg': 8.695932235019891e-06, 'mg': 6.293567636010787e-06, 'vi': 4.930251876631165e-06,
                              'rg': 4.508356206848724e-06, 'bg': 2.4407187508075073e-08, 'in': 3.4867410725821534e-09}

        # 问题分类器
        self.rough_cls_dict = {'DESCRIPTION': 0, 'ENTITY': 1, 'YES_NO': 2}
        self.fine_cls = FineClassify()

        if self.badcase_sample_log_file:
            self.badcase_dumper = open(badcase_sample_log_file, 'w')

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

        if self.badcase_sample_log_file:
            self.badcase_dumper.close()

        # 对于训练集，统计训练集的答案长度分布
        if self.train_set and self.train_answer_len_cut_bins > 0:
            self.logger.info('cut the mean answer length with same frequence')
            # 统计该样本答案的长度
            train_answer_lens = []
            for sample in self.train_set:
                ans_lens = [len(ans) for ans in sample['segmented_answers']]
                mean_ans_len = sum(ans_lens) / len(ans_lens)
                train_answer_lens.append(mean_ans_len)
                sample['mean_answer_len'] = mean_ans_len

            def same_freq_bincut(series, n):
                edages = pd.Series([i / n for i in range(n)])  # 转换成百分比
                func = lambda x: (edages >= x).values.argmax()  # 函数：(edages >= x)返回fasle/true列表中第一次出现true的索引值
                return series.rank(pct=1).astype(float).apply(func)

            sample_belong_bins = same_freq_bincut(pd.Series(train_answer_lens), self.train_answer_len_cut_bins)

            # 按照不同的 bin 进行划分训练集
            self.bin_cut_train_sets = [[] for _ in range(self.train_answer_len_cut_bins)]
            for i, sample in enumerate(self.train_set):
                answer_bin = sample_belong_bins[i]
                self.bin_cut_train_sets[answer_bin].append(sample)

            self.min_bin_data_size = min([len(bin_set) for bin_set in self.bin_cut_train_sets])
            self.bin_set_count = np.array([len(s) for s in self.bin_cut_train_sets])
            self.train_set.clear()  # save memory
            self.logger.info('bincut done.')
        else:
            self.bin_cut_train_sets = []    # dev/test

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        """
        if train and self.max_a_len is None:
            raise ValueError('must provide max_a_len for training set!')

        badcase_sample_cnt = 0  # 错误样本的数目
        with io.open(data_path, 'r', encoding='utf-8') as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                if '{' not in line:
                    continue

                sample = json.loads(line.strip())
                bad_case_sample = False

                if train:  # 仅对训练集进行 bad case 数据清洗
                    assert self.badcase_sample_log_file is not None

                    if len(sample['segmented_question']) == 0 or len(sample['documents']) == 0:
                        bad_case_sample = True
                        sample['error_info'] = 'empty_question'
                    elif len(sample['fake_answers']) == 0:
                        bad_case_sample = True
                        sample['error_info'] = 'empty_fake_answer'
                    else:
                        best_match_doc_ids = []
                        best_match_scores = []
                        answer_labels = []
                        fake_answers = []

                        # 策略一：统计答案的平均长度，如果超过 max_a_len，则过滤该样本
                        ans_len = [len(ans) for ans in sample['segmented_answers']]
                        if sum(ans_len) / len(ans_len) > self.max_a_len:
                            continue

                        for ans_idx, answer_label in enumerate(sample['answer_labels']):
                            # 对于 multi-answer 有的fake answer 没有找到
                            if answer_label[0] == -1 or answer_label[1] == -1:
                                continue

                            # # 策略二：对单个答案进行处理，单个长度超过 max_a_len，去掉这个outlier答案，
                            # # 如果去掉之后 answers 为空了则去掉整个样本，如果不为空，用第二好的answer
                            # if answer_label[1] - answer_label[0] + 1 > self.max_a_len:
                            #     continue

                            best_match_doc_ids.append(sample['best_match_doc_ids'][ans_idx])
                            best_match_scores.append(sample['best_match_scores'][ans_idx])
                            answer_labels.append(sample['answer_labels'][ans_idx])
                            fake_answers.append(sample['fake_answers'][ans_idx])

                        if len(best_match_doc_ids) == 0:
                            bad_case_sample = True
                            sample['error_info'] = 'empty_fake_answer'
                        else:
                            sample['best_match_doc_ids'] = best_match_doc_ids
                            sample['best_match_scores'] = best_match_scores
                            sample['answer_labels'] = answer_labels
                            sample['fake_answers'] = fake_answers

                if bad_case_sample:
                    badcase_sample_cnt += 1
                    self.badcase_dumper.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    self.badcase_dumper.flush()
                else:
                    data_set.append(sample)

        return data_set

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name == 'train' and self.train_answer_len_cut_bins > 0:  # 存在 bin
            data_set = []
            for bin_set in self.bin_cut_train_sets:
                data_set += bin_set
        elif set_name == 'train' and self.train_answer_len_cut_bins <= 0:  # 不存在 bin
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

        if data_set is not None:
            for sample in data_set:
                for token in sample['segmented_question']:
                    yield token
                for doc in sample['documents']:
                    for token in doc['segmented_passage']:
                        yield token

    def convert_to_ids(self, vocab, use_oov2unk):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
            use_oov2unk: 所有oov的词是否映射到 <unk>, 默认为 False
        """
        # 如果是train, 则丢弃segmented_passage字段
        if self.train_answer_len_cut_bins > 0:  # 存在 bin
            for bin_set in self.bin_cut_train_sets:
                for sample in bin_set:
                    sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
                    for doc in sample['documents']:
                        doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)
                        doc['segmented_passage'] = []
        elif self.train_set:
            for sample in self.train_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
                for doc in sample['documents']:
                    doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)

        for data_set in [self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
                for doc in sample['documents']:
                    doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)

    def get_data_length(self, set_name):
        if set_name == 'train' and self.train_answer_len_cut_bins > 0:
            return sum([len(bin_set) for bin_set in self.bin_cut_train_sets])
        elif set_name == 'train' and self.train_answer_len_cut_bins <= 0:
            return len(self.train_set)
        elif set_name == 'dev':
            return len(self.dev_set)
        elif set_name == 'test':
            return len(self.test_set)
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

    def get_real_batch_size(self, batch_size, set_name):
        """ 获取实际的batch_size大小 """
        if set_name == 'train' and self.train_answer_len_cut_bins > 0:
            bin_batch_size = batch_size // self.train_answer_len_cut_bins
            real_batch_size = bin_batch_size * self.train_answer_len_cut_bins
            return real_batch_size
        elif set_name == 'train' and self.train_answer_len_cut_bins <= 0:
            return batch_size
        elif set_name == 'dev' or set_name == 'test':
            return batch_size
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train' and self.train_answer_len_cut_bins > 0:
            # 分别对每个 bin 的数据进行 shuffle 再进行均衡采样
            data_size = self.min_bin_data_size
            if shuffle:
                for bin_set in self.bin_cut_train_sets:
                    np.random.shuffle(bin_set)

            bin_left_idx = [0] * self.train_answer_len_cut_bins
            bin_batch_size = batch_size // self.train_answer_len_cut_bins
            # 实际训练的 batch size 大小
            real_batch_size = bin_batch_size * self.train_answer_len_cut_bins
            for batch_start in np.arange(0, data_size, bin_batch_size):
                # 从每个 bin_set 中均衡采样数据
                batch_set = []
                should_concat = False
                for bin_i, bin_set in enumerate(self.bin_cut_train_sets):
                    batch_binset = bin_set[batch_start: batch_start + bin_batch_size]
                    if len(batch_binset) < bin_batch_size:
                        should_concat = True
                        break
                    batch_set += batch_binset
                if not should_concat:
                    for bin_i, bin_set in enumerate(self.bin_cut_train_sets):
                        bin_left_idx[bin_i] = batch_start + bin_batch_size
                    yield self._one_mini_batch(batch_set, range(len(batch_set)), pad_id, is_testing=False)
                else:
                    break

            # 剩下的bin的样本进行拼接，同时将数目最小的bin进行适当的数据上采样
            bin_left_cnts = list(self.bin_set_count - np.array(bin_left_idx))
            least_bin_cnt = min(bin_left_cnts)
            least_bin_idx = bin_left_cnts.index(least_bin_cnt)
            del bin_left_cnts[least_bin_idx]
            second_least_bin_cnt = min(bin_left_cnts)
            # least_bin_idx 的 bin 上采样的数目 upsample_cnt
            upsample_cnt = second_least_bin_cnt - least_bin_cnt

            left_set = []
            for idx, bin_set in enumerate(self.bin_cut_train_sets):
                if idx == least_bin_idx:
                    sample_idxs = [randint(0, self.bin_set_count[idx]) for _ in range(0, upsample_cnt)]
                    upsample = map(bin_set.__getitem__, sample_idxs)
                    bin_set += upsample
                left_set += bin_set[bin_left_idx[idx]:]

            if len(left_set) > 0:
                if shuffle:
                    np.random.shuffle(left_set)

                left_data_size = len(left_set)
                for batch_start in np.arange(0, left_data_size, real_batch_size):
                    left_batch_set = left_set[batch_start: batch_start + real_batch_size]
                    if len(left_batch_set) < real_batch_size:
                        break
                    yield self._one_mini_batch(left_batch_set, range(len(left_batch_set)), pad_id, is_testing=False)
        else:
            if set_name == 'train' and self.train_answer_len_cut_bins <= 0:
                data_set = self.train_set
                is_testing = False
            elif set_name == 'dev':
                data_set = self.dev_set
                is_testing = True
            elif set_name == 'test':
                data_set = self.test_set
                is_testing = True
            else:
                raise NotImplementedError('No data set named as {}'.format(set_name))

            data_size = len(data_set)
            indices = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices)
            for batch_start in np.arange(0, data_size, batch_size):
                batch_indices = indices[batch_start: batch_start + batch_size]
                yield self._one_mini_batch(data_set, batch_indices, pad_id, is_testing=is_testing)

    def _split_list_by_specific_value(self, iterable, splitters):
        return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]

    def _one_mini_batch(self, data, indices, pad_id, is_testing):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_raw_data = []
        for i in indices:
            sample = data[i]
            cleaned_sample = {'documents': [{'segmented_passage': doc['segmented_passage']} for doc in sample['documents']],
                              'question_id': sample['question_id'],
                              'question_type': sample['question_type'],
                              'segmented_question': sample['segmented_question']}
            if 'segmented_answers' in sample:
                cleaned_sample['segmented_answers'] = sample['segmented_answers']
            if 'answer_labels' in sample:
                cleaned_sample['answer_labels'] = sample['answer_labels']

            batch_raw_data.append(cleaned_sample)

        batch_data = {'raw_data': batch_raw_data,
                      'question_token_ids': [],
                      'pos_questions': [],
                      'pos_freq_questions': [],
                      'keyword_questions': [],
                      'question_length': [],
                      'question_rough_cls': [],
                      'question_fine_cls': [],

                      'passage_token_ids': [],
                      'pos_passages': [],
                      'pos_freq_passages': [],
                      'keyword_passages': [],
                      'passage_length': [],

                      'wiq_feature': [],
                      'passage_para_match_socre': [],
                      'doc_ids': [],    # doc 的位置编码，所在下标

                      # 距离特征
                      'para_count_based_cos_distance': [],
                      'para_levenshtein_distance': [],
                      'para_fuzzy_matching_ratio': [],
                      'para_fuzzy_matching_partial_ratio': [],
                      'para_fuzzy_matching_token_sort_ratio': [],
                      'para_fuzzy_matching_token_set_ratio': [],

                      'start_ids': [],
                      'end_ids': [],
                      'is_selected': [],
                      'match_scores': []}

        batch_samples = [data[i] for i in indices]

        max_passage_num = max([len(sample['documents']) for sample in batch_samples])
        max_passage_num = min(self.max_p_num, max_passage_num)
        # 增加信息,求最大答案数
        if not is_testing:
            max_ans_num = max([len(sample['answer_labels']) for sample in batch_samples])
        else:
            max_ans_num = 1

        for sidx, sample in enumerate(batch_samples):
            for pidx in range(max_passage_num):
                if pidx < len(sample['documents']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    # question 分类信息
                    batch_data['question_rough_cls'].append([self.rough_cls_dict[sample['question_type']]] * len(sample['question_token_ids']))
                    question_str = ''.join(sample['segmented_question'])
                    batch_data['question_fine_cls'].append([self.fine_cls.get_classify_label(question_str)[0]] * len(sample['question_token_ids']))

                    passage_token_ids = sample['documents'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                    batch_data['pos_questions'].append([self.pos_meta_dict[pos_str] if pos_str in self.pos_meta_dict else self.pos_meta_dict['other'] for pos_str in sample['pos_question']])
                    batch_data['pos_freq_questions'].append([self.pos_freq_dict[pos_str] for pos_str in sample['pos_question']])
                    batch_data['keyword_questions'].append(sample['keyword_question'])
                    batch_data['pos_passages'].append([self.pos_meta_dict[pos_str] if pos_str in self.pos_meta_dict else self.pos_meta_dict['other'] for pos_str in sample['documents'][pidx]['pos_passage']])
                    batch_data['pos_freq_passages'].append([self.pos_freq_dict[pos_str] for pos_str in sample['documents'][pidx]['pos_passage']])
                    batch_data['keyword_passages'].append(sample['documents'][pidx]['keyword_passage'])
                    batch_data['wiq_feature'].append(sample['documents'][pidx]['passage_word_in_question'])
                    batch_data['doc_ids'].append([pidx] * len(passage_token_ids))

                    # 1. paragraph 和 question 的 max_f1 * bleu
                    para_match_socre = []
                    # 2. count-based cos-distance
                    para_count_based_cos_distance = []
                    # 3. levenshtein_distance
                    para_levenshtein_distance = []
                    # 4. fuzzy_matching_ratio
                    para_fuzzy_matching_ratio = []
                    para_fuzzy_matching_partial_ratio = []
                    para_fuzzy_matching_token_sort_ratio = []
                    para_fuzzy_matching_token_set_ratio = []

                    doc = sample['documents'][pidx]
                    paras = self._split_list_by_specific_value(doc['segmented_passage'], ('<splitter>',))
                    for para_i, para in enumerate(paras):
                        para_match_socre.extend([doc['paragraph_match_score'][para_i]] * len(para) + [0])
                        para_count_based_cos_distance.extend([doc['para_count_based_cos_distance'][para_i]] * len(para) + [0])
                        para_levenshtein_distance.extend([doc['para_levenshtein_distance'][para_i]] * len(para) + [0])
                        para_fuzzy_matching_ratio.extend([doc['para_fuzzy_matching_ratio'][para_i]] * len(para) + [0])
                        para_fuzzy_matching_partial_ratio.extend([doc['para_fuzzy_matching_partial_ratio'][para_i]] * len(para) + [0])
                        para_fuzzy_matching_token_sort_ratio.extend([doc['para_fuzzy_matching_token_sort_ratio'][para_i]] * len(para) + [0])
                        para_fuzzy_matching_token_set_ratio.extend([doc['para_fuzzy_matching_token_set_ratio'][para_i]] * len(para) + [0])

                    batch_data['passage_para_match_socre'].append(para_match_socre[:-1])
                    batch_data['para_count_based_cos_distance'].append(para_count_based_cos_distance[:-1])
                    batch_data['para_levenshtein_distance'].append(para_levenshtein_distance[:-1])
                    batch_data['para_fuzzy_matching_ratio'].append(para_fuzzy_matching_ratio[:-1])
                    batch_data['para_fuzzy_matching_partial_ratio'].append(para_fuzzy_matching_partial_ratio[:-1])
                    batch_data['para_fuzzy_matching_token_sort_ratio'].append(para_fuzzy_matching_token_sort_ratio[:-1])
                    batch_data['para_fuzzy_matching_token_set_ratio'].append(para_fuzzy_matching_token_set_ratio[:-1])

                    if not is_testing:
                        batch_data['is_selected'].append(
                            int(sample['documents'][pidx]['is_selected']))
                    else:
                        batch_data['is_selected'].append(0)
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
                    # 增加信息
                    batch_data['question_rough_cls'].append([])
                    batch_data['question_fine_cls'].append([])
                    batch_data['pos_questions'].append([])
                    batch_data['pos_freq_questions'].append([])
                    batch_data['keyword_questions'].append([])
                    batch_data['pos_passages'].append([])
                    batch_data['pos_freq_passages'].append([])
                    batch_data['keyword_passages'].append([])
                    batch_data['wiq_feature'].append([])
                    batch_data['doc_ids'].append([])

                    batch_data['passage_para_match_socre'].append([])
                    batch_data['para_count_based_cos_distance'].append([])
                    batch_data['para_levenshtein_distance'].append([])
                    batch_data['para_fuzzy_matching_ratio'].append([])
                    batch_data['para_fuzzy_matching_partial_ratio'].append([])
                    batch_data['para_fuzzy_matching_token_sort_ratio'].append([])
                    batch_data['para_fuzzy_matching_token_set_ratio'].append([])
                    batch_data['is_selected'].append(0)

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)

        # 增加信息,修改
        if not is_testing:
            for sample in batch_samples:
                start_ids = []
                end_ids = []
                scores = []
                for aidx in range(max_ans_num):
                    if aidx < len(sample['best_match_doc_ids']):
                        gold_passage_offset = padded_p_len * sample['best_match_doc_ids'][aidx]
                        start_ids.append(gold_passage_offset + sample['answer_labels'][aidx][0])
                        end_ids.append(gold_passage_offset + sample['answer_labels'][aidx][1])
                        scores.append(sample['best_match_scores'][aidx])
                    else:
                        start_ids.append(0)
                        end_ids.append(0)
                        scores.append(0)
                batch_data['start_ids'].append(start_ids)
                batch_data['end_ids'].append(end_ids)
                batch_data['match_scores'].append(scores)
        else:
            # test阶段
            batch_data['start_ids'] = [[0] * max_ans_num] * len(indices)
            batch_data['end_ids'] = [[0] * max_ans_num] * len(indices)
            batch_data['match_scores'] = [[0] * max_ans_num] * len(indices)

        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [
            (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len] for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [
            (ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len] for ids in batch_data['question_token_ids']]
        # 增加信息
        batch_data['pos_questions'] = [
            (pos + [-1] * (pad_q_len - len(pos)))[: pad_q_len] for pos in batch_data['pos_questions']]
        batch_data['keyword_questions'] = [
            (key + [-1] * (pad_q_len - len(key)))[: pad_q_len] for key in batch_data['keyword_questions']]
        batch_data['pos_freq_questions'] = [
            (freq + [0.0] * (pad_q_len - len(freq)))[: pad_q_len] for freq in batch_data['pos_freq_questions']]
        batch_data['question_rough_cls'] = [
            (rough + [-1] * (pad_q_len - len(rough)))[: pad_q_len] for rough in batch_data['question_rough_cls']]
        batch_data['question_fine_cls'] = [
            (fine + [-1] * (pad_q_len - len(fine)))[: pad_q_len] for fine in batch_data['question_fine_cls']]

        batch_data['pos_passages'] = [
            (pos + [-1] * (pad_p_len - len(pos)))[: pad_p_len] for pos in batch_data['pos_passages']]
        batch_data['keyword_passages'] = [
            (key + [-1] * (pad_p_len - len(key)))[: pad_p_len] for key in batch_data['keyword_passages']]
        batch_data['pos_freq_passages'] = [
            (freq + [0.0] * (pad_p_len - len(freq)))[: pad_p_len] for freq in batch_data['pos_freq_passages']]

        batch_data['wiq_feature'] = [(wiq + [-1] * (pad_p_len - len(wiq)))[: pad_p_len] for wiq in batch_data['wiq_feature']]
        batch_data['passage_para_match_socre'] = [(wiq + [0] * (pad_p_len - len(wiq)))[: pad_p_len] for wiq in batch_data['passage_para_match_socre']]

        batch_data['para_count_based_cos_distance'] = [(dist + [0] * (pad_p_len - len(dist)))[: pad_p_len] for dist in batch_data['para_count_based_cos_distance']]
        batch_data['para_levenshtein_distance'] = [(dist + [0] * (pad_p_len - len(dist)))[: pad_p_len] for dist in batch_data['para_levenshtein_distance']]
        batch_data['para_fuzzy_matching_ratio'] = [(dist + [0] * (pad_p_len - len(dist)))[: pad_p_len] for dist in batch_data['para_fuzzy_matching_ratio']]
        batch_data['para_fuzzy_matching_partial_ratio'] = [(dist + [0] * (pad_p_len - len(dist)))[: pad_p_len] for dist in batch_data['para_fuzzy_matching_partial_ratio']]
        batch_data['para_fuzzy_matching_token_sort_ratio'] = [(dist + [0] * (pad_p_len - len(dist)))[: pad_p_len] for dist in batch_data['para_fuzzy_matching_token_sort_ratio']]
        batch_data['para_fuzzy_matching_token_set_ratio'] = [(dist + [0] * (pad_p_len - len(dist)))[: pad_p_len] for dist in batch_data['para_fuzzy_matching_token_set_ratio']]

        batch_data['doc_ids'] = [(did + [-1] * (pad_p_len - len(did)))[: pad_p_len] for did in batch_data['doc_ids']]

        return batch_data, pad_p_len, pad_q_len
