#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/5 16:49
"""
import re
import sys
import json
import collections


def gen_trainable_dataset(sample):
    pass


if __name__ == '__main__':
    for line in sys.stdin:
        sample = json.loads(line.strip())
        gen_trainable_dataset(sample)
        print(json.dumps(sample, ensure_ascii=False))
