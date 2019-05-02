#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/1 22:46
"""
import sys
import json
from tqdm import tqdm

for line in sys.stdin:
    if not line.startswith('{'):
        continue

    sample = json.loads(line.strip())
    question_id = sample['question_id']

    rouge_l, bleu4 = sample['ceil_rouge_l'], sample['ceil_bleu4']

    if rouge_l > 60:
        print(line)
