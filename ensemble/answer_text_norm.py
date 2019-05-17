#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/17 23:06
"""
import pandas as pd


class AnswerNormer(object):

    def __init__(self, url_map_path, ):
        url_fra = pd.read_csv(url_map_path)
        # 建立反向url_map表
        self.idx2url = {}
        for url, idx in zip(url_fra['url'], url_fra['url_map_id']):
            self.idx2url[idx] = url

        self.spaces = {'\x10', '\x7f', '\x9d', '\xad', '\\x0a', '\\xa0', '\\x0d',
                       '\f', '\n', '\r', '\t', '\v', '&#160;', '&nbsp;',
                       '\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u1680', '\u180e',
                       '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
                       '\u2009', '\u200a', '\u2028', '\u2029', '\u202f', '\u205f', '\u3000'}

    def _remove_space(self, text):
        for space in self.spaces:
            text = text.replace(space, '')
        text = text.strip()
        return text

    def _convert_punctuation(self, text):
        text = text.replace('<splitter>', '')
        return text

    def norm_predict_answer(self, seg_answer):
        # 开头的标点符号的去除
        start = 0
        while start < len(seg_answer):
            if seg_answer[start] not in {'.', ',', ';', ')'}:
                break
            start += 1
        seg_ans = seg_answer[start:]

        # url 映射处理
        for i, token in enumerate(seg_ans):
            if token in self.idx2url:
                seg_ans[i] = self.idx2url[token]
                # seg_ans[i] = ''  # 删除url_xxx模式

        # 拼接list为str
        pre_ans = ''.join(seg_ans).replace('......', '').replace('谢邀.', '').replace('谢邀,', '') \
            .replace(',,', ',').replace(';;', ';').replace('!.', '.') \
            .replace('哈哈,', '').replace('哈哈.', '').replace('哈哈!', '').replace('哈哈', '') \
            .replace('呵呵,', '').replace('呵呵.', '').replace('呵呵!', '').replace('哈哈', '') \
            .replace('是是', '').replace('的的', '的')

        # 字符串替换处理
        post_ans = self._remove_space(pre_ans)
        # spliter 去除
        post_ans = self._convert_punctuation(post_ans)
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
