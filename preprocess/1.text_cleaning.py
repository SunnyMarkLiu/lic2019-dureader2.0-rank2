#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/5 01:07
"""
import sys
sys.path.append('../')

import re
import sys
import json
import jieba
import collections
import pandas as pd
from utils.jieba_util import WordSegmentPOSKeywordExtractor
import warnings
warnings.filterwarnings("ignore")

data_version = 'dureader_2.0_v3'

jieba.load_userdict(f'../input/{data_version}/jieba_custom_dict.txt')      # 抽取的entity和所有的 url 构成的词
url_map_df = pd.read_csv(f'../input/{data_version}/url_mapping.csv', encoding='utf-8')
url_map_dict = dict(zip(url_map_df['url'], url_map_df['url_map_id']))
jieba_extractor = WordSegmentPOSKeywordExtractor()
print('jieba prepared.')

# remove space
spaces = {'\x10', '\x7f', '\x9d', '\xad', '\\x0a', '\\xa0', '\\x0d'
          '\f', '\n', '\r', '\t', '\v', '&#160;', '&nbsp;',
          '\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u1680', '\u180e',
          '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
          '\u2009', '\u200a', '\u2028', '\u2029', '\u202f', '\u205f', '\u3000'}

def _remove_space(text):
    for space in spaces:
        text = text.replace(space, '')
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text


error_word_map = {
    '唿': '呼'
}

def _clean_error_word(text):
    """
    错别字清洗，如 '唿'吸道,打招'唿'
    """
    for error_word in error_word_map:
        if error_word in text:
            text = text.replace(error_word, error_word_map[error_word])
    return text


def _remove_html_tag(text):
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, '', text)
    return text


remove_regx_map = collections.OrderedDict({
    r'\s+': ' ',
    r'<(\d+)>': '\g<1>',
    r'\^[A-Z]': '',  # '^G', '^H', '^E'去除
    r'步骤阅读': '',  # 注意需要针对具体的 case，分析模型预测的答案和实际的答案的差别来进行相应字段的清洗
    r'(\!|\"|\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\\|\]|\^|\_|\`|\{|\||\}|\~)\1{1,}': '\g<1>',
    r'("""|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|　|、|〃|〈|〉|《|》|'
    r'「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏|﹑|﹔|·|！|？|｡|。)\1{1,}': '\g<1>',
})


def _remove_by_regex(text):
    text = text.strip()

    for rgx in remove_regx_map:
        text = re.sub(rgx, remove_regx_map[rgx], text)

    return text


def _url_replace(text):
    """ url 链接替换，注意答案中也存在 url，预测完答案后再 mapping 回来 """
    for url in url_map_dict:
        if url in text:
            text = text.replace(url, url_map_dict[url])
    return text


def _clean_duplacte_words(text):
    """
    去除很多重复的词和标点符号
    """
    reg = r'([^0-9IX]+)(\1){2,}'
    for i in range(4):
        temp = text
        text = re.sub(reg, lambda m: m.group(1), text)
        if len(text) == len(temp):
            break
    return text


def clean_document(document, answers=None):
    title = document['title']
    paragraphs = document['paragraphs']

    # --------------------- clean title ---------------------
    title = _remove_html_tag(title)
    title = _remove_by_regex(title)
    # remove website name
    if '_' in title:
        title = ''.join(title.split('_')[:-1])
    elif '-' in title:
        title = ''.join(title.split('-')[:-1])

    # --------------------- clean paragraphs ---------------------
    # ans_has_html = re.match('<[a-zA-Z]+>', ''.join(answers), flags=0) is not None

    new_paragraphs = []
    for paragraph in paragraphs:
        # 大量url链接的清洗
        paragraph = paragraph.replace('http：//', 'http://')
        paragraph = paragraph.replace('https：//', 'https://')
        paragraph = _url_replace(paragraph)
        # 如果答案包含标签则不清洗html标签
        # if not ans_has_html:
        paragraph = _remove_html_tag(paragraph)
        # 错误词的纠正
        paragraph = _clean_error_word(paragraph)
        # 按照正则表达式去除特定文本
        paragraph = _remove_by_regex(paragraph)
        # 去除空格
        paragraph = _remove_space(paragraph)
        # 去除重复的词
        paragraph = _clean_duplacte_words(paragraph)

        # 去除空段落和重复段落
        if paragraph != '' and paragraph not in new_paragraphs:
            new_paragraphs.append(paragraph)

    document['title'] = _remove_space(title)
    document['paragraphs'] = new_paragraphs

    return document


def clean_sample(sample):
    question = sample['question'].strip()
    documents = sample['documents']

    if 'answers' in sample:  # 只对 train/dev 数据进行过滤
        if not question or not len(documents):
            return False

        if not len(sample['answers']):
            return False

    documents = [clean_document(document) for document in documents]

    sample['documents'] = documents
    return True


def _nlp_text_analyse(sample):
    """ 对问题和文章进行中文分词，词性标注和关键词抽取等 """
    # question
    sample['segmented_question'], sample['pos_question'], sample['keyword_question'] = \
        jieba_extractor.extract_sentence(sample['question'], keyword_ratios=0.6)

    question_sapns = set(sample['segmented_question'])

    # documents
    new_documents = []
    for document in sample['documents']:
        if len(document['paragraphs']) == 0:
            continue

        document['segmented_title'], document['pos_title'], document['keyword_title'] = [], [], []
        if document['title'] != '':
            document['segmented_title'], document['pos_title'], document['keyword_title'] = \
                jieba_extractor.extract_sentence(document['title'], keyword_ratios=0.6)

        document['title_word_in_question'] = [int(token in question_sapns) for token in document['segmented_title']]

        document['segmented_paragraphs'], document['pos_paragraphs'], document['keyword_paragraphs'] = [], [], []
        document['paragraphs_word_in_question'] = []
        for para in document['paragraphs']:
            seg_para, pos_para, keyword_para = jieba_extractor.extract_sentence(para, keyword_ratios=0.4)
            document['segmented_paragraphs'].append(seg_para)
            document['pos_paragraphs'].append(pos_para)
            document['keyword_paragraphs'].append(keyword_para)
            document['paragraphs_word_in_question'].append([int(token in question_sapns) for token in seg_para])

        new_documents.append(document)
    sample['documents'] = new_documents

    # answer
    if 'answers' in sample:
        sample['segmented_answers'] = [list(jieba_extractor.extract_sentence(answer)) for answer in sample['answers']]


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        if clean_sample(sample):
            _nlp_text_analyse(sample)
            print(json.dumps(sample, ensure_ascii=False))
