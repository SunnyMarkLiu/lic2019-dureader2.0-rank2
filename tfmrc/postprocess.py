#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/4 22:51
"""
import json
import pandas as pd

# ------------------- url mapping -------------------
map_path = '../input/dureader_2.0_v5/url_mapping.csv'
url_fra = pd.read_csv(map_path)
# 建立反向url_map表
idx2url = {}
for url, idx in zip(url_fra['url'], url_fra['url_map_id']):
    idx2url[idx] = url

# remove space
spaces = {'\x10', '\x7f', '\x9d', '\xad', '\\x0a', '\\xa0', '\\x0d',
          '\f', '\n', '\r', '\t', '\v', '&#160;', '&nbsp;',
          '\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u1680', '\u180e',
          '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
          '\u2009', '\u200a', '\u2028', '\u2029', '\u202f', '\u205f', '\u3000'}


def _remove_space(text):
    for space in spaces:
        text = text.replace(space, '')
    text = text.strip()
    return text


def _convert_punctuation(text):
    text = text.replace('<splitter>', '')
    return text


def post_proc(pre_ans_list):
    # url 映射处理
    for i, token in enumerate(pre_ans_list):
        if token in idx2url:
            pre_ans_list[i] = idx2url[token]
    #             pre_ans_list[i] = ''  # 删除url_xxx模式

    # 拼接list为str
    pre_ans = ''.join(pre_ans_list)

    # 字符串替换处理
    post_ans = _remove_space(pre_ans)
    post_ans = _convert_punctuation(post_ans)
    # 末尾添加句号
    # TODO 存在bug
    if len(post_ans) >= 10 and post_ans[-1] != '。' and post_ans[-1] != '.' and post_ans[-1] != '！' and post_ans[
        -1] != '!':
        post_ans += '。'

    return post_ans

# 建立id2yesno字典
id2yesno = {}
int2str = {0: 'Yes', 1: 'No', 2: 'Depends'}
with open('yesno模型预测结果.json') as fin:
    for line in fin.readlines():
        sample = json.loads(line.strip())
        id2yesno[int(sample['question_id'])] = int2str[int(sample['yesno_pred'])]

# ------------------- yesno answer -------------------
# 建立id2yesno字典
id2yesno = {}
int2str = {0: 'Yes', 1: 'No', 2: 'Depends'}
with open('yesno模型预测结果.json') as fin:
    for line in fin.readlines():
        sample = json.loads(line.strip())
        id2yesno[int(sample['question_id'])] = int2str[int(sample['yesno_pred'])]

with open('cache/results/search/test.predicted.json', 'r') as fin:
    with open('search_postprocess_yesno.json', 'w') as fout:
        for idx, line in enumerate(fin.readlines()):
            if idx % 2000 == 0:
                print(idx)
            sample = line.strip()
            sample = json.loads(sample)
            sample['answers'][0] = post_proc(sample['segmented_answers'])
            del sample['segmented_answers']
            del sample['segmented_question']

            if sample['question_type'] == 'YES_NO':
                sample['yesno_answers'] = [id2yesno[sample['question_id']]]
            if 'segmented_question' in sample:
                del sample['segmented_question']
            if 'segmented_answers' in sample:
                del sample['segmented_answers']

            content = json.dumps(sample, ensure_ascii=False)
            fout.write(content + '\n')
