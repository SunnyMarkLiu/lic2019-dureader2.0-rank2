#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/15 21:11
"""
import io
import json
import logging
import numpy as np


class EnsembleDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    id 转换和 dynamic pooling提前做好存储在数组类型的 dataset 中
    """

    def __init__(self,
                 max_p_num,
                 max_p_len,
                 max_q_len,
                 max_a_len,
                 test_file,
                 predicted_test_files):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

        self.test_set = self._load_dataset(test_file, predicted_test_files)

    def get_data_length(self):
        return len(self.test_set)

    def _load_dataset(self, data_path, predict_test_files):
        """
        Loads the dataset
        """
        self.logger.info('load {} model predicted results'.format(len(predict_test_files)))
        predict_start_probs = {}    # 记录该样本所有模型预测的概率
        predict_end_probs = {}

        for predict_file in predict_test_files:
            with open(predict_file, 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    question_id = sample['question_id']
                    start_prob = sample['start_prob']
                    end_prob = sample['end_prob']

                    if question_id not in predict_start_probs:
                        predict_start_probs[question_id] = [start_prob]
                    else:
                        predict_start_probs[question_id].append(start_prob)

                    if question_id not in predict_end_probs:
                        predict_end_probs[question_id] = [end_prob]
                    else:
                        predict_end_probs[question_id].append(end_prob)

        with io.open(data_path, 'r', encoding='utf-8') as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                if '{' not in line:
                    continue

                sample = json.loads(line.strip())
                question_id = sample['question_id']
                sample['start_prob'] = predict_start_probs[question_id]
                sample['end_prob'] = predict_end_probs[question_id]
                data_set.append(sample)
        return data_set

    def gen_test_mini_batches(self, batch_size):
        data_size = len(self.test_set)
        indices = np.arange(data_size)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(self.test_set, batch_indices)

    def _one_mini_batch(self, data, indices):
        batch_raw_data = []
        start_probs = []
        end_probs = []
        for i in indices:
            sample = data[i]
            cleaned_sample = {'documents': [{'segmented_passage': doc['segmented_passage']} for doc in sample['documents']],
                              'question_id': sample['question_id'],
                              'question_type': sample['question_type'],
                              'segmented_question': sample['segmented_question']}
            if 'segmented_answers' in sample:
                cleaned_sample['segmented_answers'] = sample['segmented_answers']

            batch_raw_data.append(cleaned_sample)
            start_probs.append(sample['start_prob'])
            end_probs.append(sample['end_prob'])

        batch_data = {'raw_data': batch_raw_data,
                      'passage_length': [],
                      'start_probs': start_probs,
                      'end_probs': end_probs}

        batch_samples = [data[i] for i in indices]

        max_passage_num = max([len(sample['documents']) for sample in batch_samples])
        max_passage_num = min(self.max_p_num, max_passage_num)

        for sidx, sample in enumerate(batch_samples):
            for pidx in range(max_passage_num):
                if pidx < len(sample['documents']):
                    segmented_passage = sample['documents'][pidx]['segmented_passage']
                    batch_data['passage_length'].append(min(len(segmented_passage), self.max_p_len))
                else:
                    batch_data['passage_length'].append(0)

        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        batch_data['pad_p_len'] = pad_p_len

        return batch_data
