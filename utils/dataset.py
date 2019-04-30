"""
This module implements data process strategies.
"""
import io
import json
import logging
import numpy as np
from tqdm import tqdm


class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    id 转换和 dynamic pooling提前做好存储在数组类型的 dataset 中
    """

    def __init__(self,
                 max_p_num,
                 max_p_len,
                 max_q_len,
                 is_train,
                 data_files: list,
                 badcase_sample_log_file=None):
        self.logger = logging.getLogger("brc")

        self.is_train = is_train
        self.pad_id = None
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.badcase_sample_log_file = badcase_sample_log_file

        self.pos_meta_dict = {'nrt': 0, 'eng': 1, 'n': 2, 'f': 3, 'yg': 4, 'nt': 5, 'rr': 6, 'ad': 7, 'nr': 8, 'dg': 9,
                              't': 10, 'bg': 11, 'ag': 12, '<splitter>': 13, 'ns': 14, 'an': 15, 'b': 16, 'm': 17,
                              'v': 18, 'x': 19, 'q': 20, 'tg': 21, 'nz': 22, 'mq': 23, 'nrfg': 24, 'a': 25, 'i': 26,
                              'mg': 27, 's': 28, 'other': 29}

        # TODO：需要更新到 V4
        self.pos_freq_dict = {'x': 0.2505017302073833, 'n': 0.18994903727273268, 'v': 0.16773721703961722,
                               'd': 0.0442784355204184, 'uj': 0.04332142757777951, 'm': 0.035914873128451215,
                               'r': 0.03346151547125182, 'c': 0.031976904323953316, 'p': 0.027078188754676488,
                               'a': 0.024810672892150696, '<splitter>': 0.01796373223021083, 'f': 0.016534266534057693,
                               'vn': 0.015390796076855378, 'nr': 0.013331917511872515, 'ns': 0.010129986170360854,
                               'eng': 0.008881031039173504, 'l': 0.007784181286592452, 't': 0.007298874051247052,
                               'ul': 0.0072784570077686055, 'nz': 0.0052666023252793594, 'b': 0.005031279873594977,
                               'q': 0.00470416339222422, 'u': 0.004144030404554131, 'i': 0.00316889976158091,
                               'zg': 0.0031453530136113083, 'y': 0.003009212318526159, 'ad': 0.002458358223040691,
                               's': 0.0024053794903892024, 'j': 0.002053060041155933, 'ng': 0.001925258456739989,
                               'nrt': 0.0011747123914694476, 'nt': 0.0010768387875351928, 'z': 0.0008686800824306732,
                               'ug': 0.0007016803270291165, 'df': 0.0006928713256022033, 'an': 0.000656359073392133,
                               'k': 0.0005907425998064792, 'vg': 0.0005743950423346048, 'uz': 0.00046931934734238955,
                               'ud': 0.00045745315543014, 'uv': 0.00036526464955210825, 'g': 0.0002628371660524092,
                               'mq': 0.0001896734347410959, 'e': 0.00015655193744313017, 'ag': 0.00014606641220167766,
                               'o': 0.00012708224547821803, 'tg': 0.00012089244994149339, 'h': 0.00012003388411259437,
                               'nrfg': 0.00011284629585579788, 'vd': 7.992203665366032e-05, 'yg': 5.1061462337629245e-05,
                               'rz': 1.89609622415975e-05, 'rr': 1.5042305367129362e-05, 'vq': 1.2408596675776979e-05,
                               'dg': 8.066458007324866e-06, 'mg': 5.830126067861554e-06, 'rg': 4.88454343197953e-06,
                               'vi': 4.664100854289243e-06, 'bg': 1.4502801163834711e-08, 'in': 2.900560232766942e-09}

        if self.badcase_sample_log_file:
            self.badcase_dumper = open(badcase_sample_log_file, 'w')

        self.data = []
        if data_files:
            for data_file in data_files:
                self.data += self._load_dataset(data_file, train=self.is_train)
            self.logger.info('Fetch {} questions.'.format(len(self.data)))

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

    def word_iter(self):
        """
        Iterates over all the words in the dataset
        Returns:
            a generator
        """
        for sample in tqdm(self.data):
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
        for sample in tqdm(self.data):
            sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
            for doc in sample['documents']:
                doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)

    def gen_mini_batches(self, batch_size, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            batch_size: number of samples in one batch
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        data_size = len(self.data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(self.data, batch_indices)

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'pos_questions': [],
                      'pos_freq_questions': [],
                      'keyword_questions': [],
                      'question_length': [],

                      'passage_token_ids': [],
                      'pos_passages': [],
                      'pos_freq_passages': [],
                      'keyword_passages': [],
                      'passage_length': [],

                      'wiq_feature': [],

                      'start_ids': [],
                      'end_ids': [],
                      'is_selected': [],
                      'match_scores': []}

        max_passage_num = max([len(sample['documents']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)

        # 该 batch 数据中padding到的doc数
        batch_data['passage_cnts'] = [max_passage_num] * len(indices)

        # 增加信息,求最大答案数
        if self.is_train:
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
                    batch_data['pos_questions'].append([self.pos_meta_dict[pos_str] if pos_str in self.pos_meta_dict else self.pos_meta_dict['other'] for pos_str in sample['pos_question']])
                    batch_data['pos_freq_questions'].append([self.pos_freq_dict[pos_str] for pos_str in sample['pos_question']])
                    batch_data['keyword_questions'].append(sample['keyword_question'])
                    batch_data['pos_passages'].append([self.pos_meta_dict[pos_str] if pos_str in self.pos_meta_dict else self.pos_meta_dict['other'] for pos_str in sample['documents'][pidx]['pos_passage']])
                    batch_data['pos_freq_passages'].append([self.pos_freq_dict[pos_str] for pos_str in sample['documents'][pidx]['pos_passage']])
                    batch_data['keyword_passages'].append(sample['documents'][pidx]['keyword_passage'])
                    batch_data['wiq_feature'].append(sample['documents'][pidx]['passage_word_in_question'])

                    if self.is_train:
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
                    batch_data['pos_freq_questions'].append([])
                    batch_data['keyword_questions'].append([])
                    batch_data['pos_passages'].append([])
                    batch_data['pos_freq_passages'].append([])
                    batch_data['keyword_passages'].append([])
                    batch_data['wiq_feature'].append([])
                    batch_data['is_selected'].append(0)

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data)

        # 增加信息,修改
        if self.is_train:
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

    def _dynamic_padding(self, batch_data):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_id = self.pad_id
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len] for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len] for ids in batch_data['question_token_ids']]
        # 增加信息
        batch_data['pos_questions'] = [(pos + [-1] * (pad_q_len - len(pos)))[: pad_q_len] for pos in batch_data['pos_questions']]
        batch_data['keyword_questions'] = [(key + [-1] * (pad_q_len - len(key)))[: pad_q_len] for key in batch_data['keyword_questions']]
        batch_data['pos_freq_questions'] = [(freq + [0.0] * (pad_q_len - len(freq)))[: pad_q_len] for freq in batch_data['pos_freq_questions']]

        batch_data['pos_passages'] = [(pos + [-1] * (pad_p_len - len(pos)))[: pad_p_len] for pos in batch_data['pos_passages']]
        batch_data['keyword_passages'] = [(key + [-1] * (pad_p_len - len(key)))[: pad_p_len] for key in batch_data['keyword_passages']]
        batch_data['pos_freq_passages'] = [(freq + [0.0] * (pad_p_len - len(freq)))[: pad_p_len] for freq in batch_data['pos_freq_passages']]

        batch_data['wiq_feature'] = [(wiq + [-1] * (pad_p_len - len(wiq)))[: pad_p_len] for wiq in batch_data['wiq_feature']]

        return batch_data, pad_p_len, pad_q_len

    def update_pad_id(self, pad_id):
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)
