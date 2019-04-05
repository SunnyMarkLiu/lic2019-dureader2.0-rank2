#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
针对每个文档进行段落筛选。如果段落过长，则划分为多个句子，进行句子筛选。

对于训练集：问题 + 答案，进行段落筛选
对于验证集和测试集：只针对问题解析段落筛选

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/5 14:22
"""
import re
import sys
import json
import collections


def extract_paragraph(sample):
    pass


if __name__ == '__main__':
    for line in sys.stdin:
        sample = json.loads(line.strip())
        extract_paragraph(sample)
        print(json.dumps(sample, ensure_ascii=False))
