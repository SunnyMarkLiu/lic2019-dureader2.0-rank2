#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/4 22:51
"""
import sys
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


def post_process(pre_ans_list):

    # 开头的标点符号的去除
    start = 0
    while start < len(pre_ans_list):
        if pre_ans_list[start] not in {'.', ',', ';', ')'}:
            break
        start += 1
    seg_ans = pre_ans_list[start:]

    # url 映射处理
    for i, token in enumerate(seg_ans):
        if token in idx2url:
            seg_ans[i] = idx2url[token]
            # seg_ans[i] = ''  # 删除url_xxx模式

    # 拼接list为str
    pre_ans = ''.join(seg_ans).replace('......', '').replace('谢邀.', '').replace('谢邀,', '')\
                .replace(',,', ',').replace(';;', ';').replace('!.', '.') \
                .replace('哈哈,', '').replace('哈哈.', '').replace('哈哈!', '').replace('哈哈', '') \
                .replace('呵呵,', '').replace('呵呵.', '').replace('呵呵!', '').replace('哈哈', '')\
                .replace('是是', '').replace('的的', '的')

    # 字符串替换处理
    post_ans = _remove_space(pre_ans)
    # spliter 去除
    post_ans = _convert_punctuation(post_ans)
    # 末尾是,的替换为句号
    if post_ans == '':
        return post_ans

    if post_ans.endswith(','):
        tmp = list(post_ans)
        tmp[-1] = '.'
        post_ans = ''.join(tmp)
    # 末尾添加句号
    if post_ans[-1] not in {'。', '.', '！', '!'}:
        post_ans += '.'

    return post_ans


# ------------------- yesno answer -------------------
# 建立id2yesno字典
id2yesno = {}
int2str = {0: 'Yes', 1: 'No', 2: 'Depends'}
with open('../yesno/yesno.test1.predicted.json') as fin:
    for line in fin.readlines():
        sample = json.loads(line.strip())
        id2yesno[int(sample['question_id'])] = int2str[int(sample['yesno_pred'])]

if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        sample['answers'][0] = post_process(sample['segmented_answers'])

        if sample['question_type'] == 'YES_NO':
            sample['yesno_answers'] = [id2yesno[sample['question_id']]]
        if 'segmented_question' in sample:
            del sample['segmented_question']
        if 'segmented_answers' in sample:
            del sample['segmented_answers']

        del sample['start_prob']
        del sample['end_prob']
        print(json.dumps(sample, ensure_ascii=False))
