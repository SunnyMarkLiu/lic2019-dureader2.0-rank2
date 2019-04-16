"""
This module implements data process strategies.
"""
import io
import json
import logging
import numpy as np


class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    id 转换和 dynamic pooling提前做好存储在数组类型的 dataset 中
    """

    def __init__(self,
                 max_p_num,
                 max_p_len,
                 max_q_len,
                 train_files=[],
                 dev_files=[],
                 test_files=[],
                 badcase_sample_log_file=None):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.badcase_sample_log_file = badcase_sample_log_file

        self.pos_meta_dict = {'vg': 0, 'nrt': 1, 'eng': 2, 'vi': 3, 'p': 4, 'n': 5, 'uj': 6, 'f': 7, 'rz': 8, 'yg': 9,
                              'uz': 10, 'e': 11, 'nt': 12, 'rr': 13, 'j': 14, 'df': 15, 'ng': 16, 'ad': 17, 'nr': 18,
                              'vq': 19, 'dg': 20, 'ud': 21, 'o': 22, 'k': 23, 't': 24, 'rg': 25, 'bg': 26, 'ug': 27,
                              'ag': 28, 'g': 29, '<splitter>': 30, 'r': 31, 'vn': 32, 'ns': 33, 'an': 34, 'b': 35,
                              'in': 36, 'zg': 37, 'l': 38, 'z': 39, 'c': 40, 'm': 41, 'v': 42, 'x': 43, 'q': 44,
                              'uv': 45, 'd': 46, 'tg': 47, 'nz': 48, 'h': 49, 'mq': 50, 'nrfg': 51, 'a': 52, 'u': 53,
                              'i': 54, 'vd': 55, 'mg': 56, 's': 57, 'ul': 58, 'y': 59}

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

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        """
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
                        for ans_idx, answer_label in enumerate(sample['answer_labels']):
                            if answer_label[0] == -1 or answer_label[1] == -1:  # 对于 multi-answer 有的fake answer 没有找到
                                continue

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

        if self.badcase_sample_log_file:
            self.badcase_dumper.close()

        return data_set

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
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
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
                for doc in sample['documents']:
                    doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)

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
        is_testing = False
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
            is_testing = True
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id, is_testing)

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
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'pos_questions': [],
                      'keyword_questions': [],
                      'question_length': [],

                      'passage_token_ids': [],
                      'pos_passages': [],
                      'keyword_passages': [],
                      'passage_length': [],

                      'start_ids': [],
                      'end_ids': [],
                      'is_selected': [],
                      'match_scores': []}

        max_passage_num = max([len(sample['documents']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        # 增加信息,求最大答案数
        if not is_testing:
            max_ans_num = max([len(sample['answer_labels']) for sample in batch_data['raw_data']])
        else:
            max_ans_num = 1

        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['documents']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['documents'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                    batch_data['pos_questions'].append([self.pos_meta_dict[pos_str] for pos_str in sample['pos_question']])
                    batch_data['keyword_questions'].append(sample['keyword_question'])
                    batch_data['pos_passages'].append([self.pos_meta_dict[pos_str] for pos_str in sample['documents'][pidx]['pos_passage']])
                    batch_data['keyword_passages'].append(sample['documents'][pidx]['keyword_passage'])

                    if not is_testing:
                        batch_data['is_selected'].append(int(sample['documents'][pidx]['is_selected']))
                    else:
                        batch_data['is_selected'].append(0)
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
                    # 增加信息
                    batch_data['pos_questions'].append([])
                    batch_data['keyword_questions'].append([])
                    batch_data['pos_passages'].append([])
                    batch_data['keyword_passages'].append([])
                    batch_data['is_selected'].append(0)

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)

        # 增加信息,修改
        if not is_testing:
            for sample in batch_data['raw_data']:
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
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len] for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len] for ids in batch_data['question_token_ids']]
        # 增加信息
        batch_data['pos_questions'] = [(pos + [-1] * (pad_q_len - len(pos)))[: pad_q_len] for pos in batch_data['pos_questions']]
        batch_data['keyword_questions'] = [(key + [-1] * (pad_q_len - len(key)))[: pad_q_len] for key in batch_data['keyword_questions']]

        batch_data['pos_passages'] = [(pos + [-1] * (pad_p_len - len(pos)))[: pad_p_len] for pos in batch_data['pos_passages']]
        batch_data['keyword_passages'] = [(key + [-1] * (pad_p_len - len(key)))[: pad_p_len] for key in batch_data['keyword_passages']]

        return batch_data, pad_p_len, pad_q_len
