#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/29 11:07
"""
import sys
import json

if __name__ == '__main__':
    min_ceil_rouge = int(sys.argv[1])
    min_ceil_bleu4 = int(sys.argv[2])

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        rouge_l, bleu4 = sample['ceil_rouge_l'], sample['ceil_bleu4']

        if rouge_l > min_ceil_rouge or bleu4 > min_ceil_bleu4:
            print(json.dumps(sample, ensure_ascii=False))
