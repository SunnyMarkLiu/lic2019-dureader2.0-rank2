#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
验证生成的 MRC 数据集的有效性

empty fake answer：/
百度开源的 preprocessed 数据的 empty fake answer：13099/271570

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/12 17:08
"""
import sys
import json

if __name__ == '__main__':
    empty_gold_fake_sanwer_count = 0
    process_cnt = 0

    bad_case_writer = open('empty_gold_fake_sanwer_sample.json', 'w')

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        process_cnt += 1
        gold_fake_answer = sample['fake_answers']
        if len(gold_fake_answer) == 0:
            empty_gold_fake_sanwer_count += 1
            # print(json.dumps(sample, ensure_ascii=False))
            bad_case_writer.write(json.dumps(sample, ensure_ascii=False) + '\n')
            bad_case_writer.flush()

        if process_cnt % 10 == 0:
            print('process:', process_cnt, ', empty fake answer:', empty_gold_fake_sanwer_count)
        # print(json.dumps(sample, ensure_ascii=False))
