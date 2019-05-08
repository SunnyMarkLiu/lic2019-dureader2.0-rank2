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


def split_list_by_specific_value(iterable, splitters):
    return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())

        for doc in sample['documents']:
            passage_para_match_socre = []

            paras = split_list_by_specific_value(doc['pos_passage'], ('<splitter>',))

            for para_i, para in enumerate(paras):
                passage_para_match_socre.extend([doc['paragraph_match_score'][para_i]] * len(para) + [0])
            passage_para_match_socre = passage_para_match_socre[:-1]
            doc['passage_para_match_socre'] = passage_para_match_socre

        print(json.dumps(sample, ensure_ascii=False))
