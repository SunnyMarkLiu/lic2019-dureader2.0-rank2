#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
添加 yesno 的预测结果
@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/18 18:37
"""
import sys
import json

# ------------------- yesno answer -------------------
# 建立id2yesno字典
id2yesno = {}
int2str = {0: 'Yes', 1: 'No', 2: 'Depends'}
with open('../yesno/yesno.test2_ensemble_5.18.json') as fin:
    for line in fin.readlines():
        sample = json.loads(line.strip())
        id2yesno[int(sample['question_id'])] = int2str[int(sample['yesno_pred'])]

if __name__ == '__main__':
    for line in sys.stdin:
        sample = json.loads(line.strip())
        if sample['question_type'] == 'YES_NO':
            sample['yesno_answers'] = [id2yesno[sample['question_id']]]

        del sample['segmented_question']
        print(json.dumps(sample, ensure_ascii=False))
