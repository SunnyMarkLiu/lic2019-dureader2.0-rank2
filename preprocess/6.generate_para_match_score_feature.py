#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/8 18:25
"""
import sys
import json
import itertools
from distance_util import DistanceUtil

def split_list_by_specific_value(iterable, splitters):
    return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        question = ''.join(sample['segmented_question'])

        for doc in sample['documents']:
            # 2. count-based cos-distance
            para_count_based_cos_distance = []
            # 3. levenshtein_distance
            para_levenshtein_distance = []
            # 4. fuzzy_matching_ratio
            para_fuzzy_matching_ratio = []
            para_fuzzy_matching_partial_ratio = []
            para_fuzzy_matching_token_sort_ratio = []
            para_fuzzy_matching_token_set_ratio = []

            paras = split_list_by_specific_value(doc['segmented_passage'], ('<splitter>',))
            for para_i, para in enumerate(paras):
                para_str = ''.join(para)
                para_count_based_cos_distance.append(DistanceUtil.countbased_cos_distance(para_str, question))
                para_levenshtein_distance.append(DistanceUtil.levenshtein_distance(para_str, question))
                para_fuzzy_matching_ratio.append(DistanceUtil.fuzzy_matching_ratio(para_str, question, ratio_func='ratio'))
                para_fuzzy_matching_partial_ratio.append(DistanceUtil.fuzzy_matching_ratio(para_str, question, ratio_func='partial_ratio'))
                para_fuzzy_matching_token_sort_ratio.append(DistanceUtil.fuzzy_matching_ratio(para_str, question, ratio_func='token_sort_ratio'))
                para_fuzzy_matching_token_set_ratio.append(DistanceUtil.fuzzy_matching_ratio(para_str, question, ratio_func='token_set_ratio'))
            
            dis_sum = sum(para_levenshtein_distance)
            para_levenshtein_distance = [dis / dis_sum if dis_sum > 0 else 0 for dis in para_levenshtein_distance]

            doc['para_count_based_cos_distance'] = para_count_based_cos_distance
            doc['para_levenshtein_distance'] = para_levenshtein_distance
            doc['para_fuzzy_matching_ratio'] = para_fuzzy_matching_ratio
            doc['para_fuzzy_matching_partial_ratio'] = para_fuzzy_matching_partial_ratio
            doc['para_fuzzy_matching_token_sort_ratio'] = para_fuzzy_matching_token_sort_ratio
            doc['para_fuzzy_matching_token_set_ratio'] = para_fuzzy_matching_token_set_ratio

        print(json.dumps(sample, ensure_ascii=False))
