#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
对于每个样本，最初的 gold answer 是 best_match_scores 最大的对应的 doc，
数据扩充的策略是选择其他的best_match_scores所对应的较小的doc.

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/23 09:01
"""
import sys
sys.path.append('../')

import sys
import json
import itertools
from elasticsearch import Elasticsearch
from utils.metric_util import metric_max_over_ground_truths, f1_score, bleu_4

dureader_index_name = "dureader_data_augment"

TYPE_NAME = "_doc"
ES_HOST = "127.0.0.1"
ES_USER = "sunnymarkliu"
ES_PASSWD = "liuqing;"

es_client = Elasticsearch(hosts=[ES_HOST])

def fetch_document(question, answer, source, topk=6):
    """ 根据问题和答案检索对应的文档 """
    query = ' '.join(question + answer)
    should_dic = [{"match": {'segmented_passage': {"query": query, "boost": 2}}}]
    body = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        'source': {
                            "query": source     # search/zhidao 分离
                        }
                    }
                },
                "should": should_dic
            }
        }
    }

    try:
        result = es_client.search(index=dureader_index_name, doc_type=TYPE_NAME, body=body)
    except:
        return []

    fetched_docs = result['hits']['hits']
    if len(fetched_docs) == 0:
        return []

    new_docs = [fetched_doc['_source'] for fetched_doc in fetched_docs[:topk]]
    return new_docs

def split_list_by_specific_value(iterable, splitters):
    return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]

# predefined splitter
splitter = u'<splitter>'

def augment_new_sample(sample, search_zhidao):
    # 过滤没有答案的
    if len(sample['best_match_scores']) == 0:
        return None

    augment_sample = {
        'question_id': sample['question_id'],
        'fact_or_opinion': sample['fact_or_opinion'],
        'question_type': sample['question_type'],
        'segmented_question': sample['segmented_question'],
        'pos_question': sample['pos_question'],
        'keyword_question': sample['keyword_question'],
        'documents': [None] * len(sample['documents']),
        'segmented_answers': []
    }

    original_docs = []
    for doc in sample['documents']:
        original_docs.append(' '.join(doc['segmented_passage']))  # 内容和 insert to ES 一致

    best_match_score_id = sample['best_match_scores'].index(max(sample['best_match_scores']))
    best_match_doc_id = sample['best_match_doc_ids'][best_match_score_id]

    # 只有一个答案，保留最佳答案及其对应的doc，对剩下的doc进行es检索获得
    if len(sample['best_match_scores']) == 1:
        augment_sample['segmented_answers'].append(sample['segmented_answers'][0])
        augment_sample['documents'][best_match_doc_id] = sample['documents'][best_match_doc_id]

        fetched_new_docs = fetch_document(augment_sample['segmented_question'],
                                          augment_sample['segmented_answers'][0],
                                          search_zhidao, topk=10)
        # 对检索到的新的 doc 进行过滤，去除和最佳答案对应doc相同的
        final_new_docs = []
        ques_answers = [augment_sample['segmented_question'] + augment_sample['segmented_answers'][0]]
        for new_doc in fetched_new_docs:
            if ' '.join(new_doc['segmented_passage']) == original_docs[best_match_doc_id]:
                continue
            final_new_docs.append(new_doc)

        doc_i = 0
        for new_doc in final_new_docs:
            if doc_i == len(augment_sample['documents']):
                break

            new_doc['is_selected'] = False
            new_doc['segmented_passage'] = new_doc['segmented_passage'].split(' ')
            new_doc['pos_passage'] = new_doc['pos_passage'].split(' ')
            new_doc['keyword_passage'] = new_doc['keyword_passage'].split(' ')
            new_doc['passage_word_in_question'] = new_doc['passage_word_in_question'].split(' ')

            # 计算段落和 question + second_best_answer 的匹配得分 paragraph_match_score 和 most_related_para_id
            para_tokens = split_list_by_specific_value(new_doc['segmented_passage'], (splitter,))
            paragraph_match_scores = [metric_max_over_ground_truths(f1_score, bleu_4, para_token, ques_answers)
                                      for para_token in para_tokens]
            most_related_para_id = paragraph_match_scores.index(max(paragraph_match_scores))
            new_doc['most_related_para_id'] = most_related_para_id
            new_doc['paragraph_match_score'] = paragraph_match_scores

            if doc_i != best_match_doc_id:
                augment_sample['documents'][doc_i] = new_doc

            doc_i += 1

        return augment_sample

    # -------------------- answers > 1 个的 ----------------
    # 去除原始 trainset 中匹配得分的最大值，取第二大的值作为新的 gold answer
    best_match_scores = sample['best_match_scores']
    del best_match_scores[best_match_score_id]  # 原始最好的答案匹配得分去掉

    second_best_match_score_id = sample['best_match_scores'].index(max(best_match_scores))
    second_best_match_doc_id = sample['best_match_doc_ids'][second_best_match_score_id]
    second_best_answer = sample['segmented_answers'][second_best_match_score_id]

    for i in range(len(sample['segmented_answers'])):
        if i != best_match_score_id:
            augment_sample['segmented_answers'].append(sample['segmented_answers'][i])

    ques_answers = [augment_sample['segmented_question'] + answer for answer in augment_sample['segmented_answers']]
    # 同时注意新的sample中该文档也处于 ori_best_doc 位置
    second_gold_answer_doc = sample['documents'][second_best_match_doc_id]
    second_gold_answer_doc['is_selected'] = True

    # 第二好的答案对应的 doc
    para_tokens = split_list_by_specific_value(second_gold_answer_doc['segmented_passage'], (splitter,))
    paragraph_match_scores = [metric_max_over_ground_truths(f1_score, bleu_4, para_token, ques_answers) for para_token in para_tokens]
    most_related_para_id = paragraph_match_scores.index(max(paragraph_match_scores))
    second_gold_answer_doc['most_related_para_id'] = most_related_para_id
    second_gold_answer_doc['paragraph_match_score'] = paragraph_match_scores
    # 第二好的答案对应的doc，同时不改变最好doc出现的下标
    augment_sample['documents'][best_match_doc_id] = second_gold_answer_doc

    # 去除最好的答案之后的每个答案，都参与 ES 的检索
    fetched_new_docs = {}
    for ans_i, answer in enumerate(augment_sample['segmented_answers']):
        aug_docs = fetch_document(augment_sample['segmented_question'], answer, search_zhidao, topk=10)
        fetched_new_docs[ans_i] = aug_docs

    # 对检索到的新的 doc 进行过滤，去除和 second_gold_answer_doc、original_docs[best_match_doc_id]相同的
    final_new_docs = []
    answer_start_idx = {}
    for ans_i, answer in enumerate(augment_sample['segmented_answers']):  # 为每个 answer 找到 doc 依据
        if len(final_new_docs) == len(augment_sample['documents']) - 1:
            break

        # 找到该答案对应的第一个doc，且不为第一好和第二好的doc
        i = 0
        while i < len(fetched_new_docs[ans_i]):
            doc = fetched_new_docs[ans_i][i]
            i += 1
            if ' '.join(doc['segmented_passage']) == original_docs[best_match_doc_id] or \
                ' '.join(doc['segmented_passage']) == original_docs[second_best_match_doc_id]:
                continue

            if answer != second_best_answer:
                doc['is_selected'] = True

            if doc not in final_new_docs:
                final_new_docs.append(doc)
                break

        answer_start_idx[ans_i] = i

    if len(final_new_docs) < len(augment_sample['documents']) - 1:
        # 接着找其他的 doc，默认采用从其中一个answer贪心的找
        doc_prepared = False
        for ans_i, _ in enumerate(augment_sample['segmented_answers']):
            start_i = answer_start_idx[ans_i]

            while start_i < len(fetched_new_docs[ans_i]):
                doc = fetched_new_docs[ans_i][start_i]
                doc['is_selected'] = False
                if doc not in final_new_docs:
                    final_new_docs.append(doc)
                start_i += 1

                if len(final_new_docs) == len(augment_sample['documents']) - 1:
                    doc_prepared = True

            if doc_prepared:
                break

    if len(final_new_docs) < len(augment_sample['documents']) - 1:
        return None
    else:
        doc_i = 0
        final_new_docs = final_new_docs[::-1]

        while doc_i < best_match_doc_id:
            new_doc = final_new_docs.pop()
            new_doc['segmented_passage'] = new_doc['segmented_passage'].split(' ')
            new_doc['pos_passage'] = new_doc['pos_passage'].split(' ')
            new_doc['keyword_passage'] = new_doc['keyword_passage'].split(' ')
            new_doc['passage_word_in_question'] = new_doc['passage_word_in_question'].split(' ')

            # 计算段落和 question + second_best_answer 的匹配得分 paragraph_match_score 和 most_related_para_id
            para_tokens = split_list_by_specific_value(new_doc['segmented_passage'], (splitter,))
            paragraph_match_scores = [metric_max_over_ground_truths(f1_score, bleu_4, para_token, ques_answers) for para_token in para_tokens]
            most_related_para_id = paragraph_match_scores.index(max(paragraph_match_scores))
            new_doc['most_related_para_id'] = most_related_para_id
            new_doc['paragraph_match_score'] = paragraph_match_scores
            augment_sample['documents'][doc_i] = new_doc
            doc_i += 1

        doc_i += 1

        while doc_i < len(augment_sample['documents']):
            new_doc = final_new_docs.pop()
            new_doc['segmented_passage'] = new_doc['segmented_passage'].split(' ')
            new_doc['pos_passage'] = new_doc['pos_passage'].split(' ')
            new_doc['keyword_passage'] = new_doc['keyword_passage'].split(' ')
            new_doc['passage_word_in_question'] = new_doc['passage_word_in_question'].split(' ')

            # 计算段落和 question + second_best_answer 的匹配得分 paragraph_match_score 和 most_related_para_id
            para_tokens = split_list_by_specific_value(new_doc['segmented_passage'], (splitter,))
            paragraph_match_scores = [metric_max_over_ground_truths(f1_score, bleu_4, para_token, ques_answers)
                                      for para_token in para_tokens]
            most_related_para_id = paragraph_match_scores.index(max(paragraph_match_scores))
            new_doc['most_related_para_id'] = most_related_para_id
            new_doc['paragraph_match_score'] = paragraph_match_scores

            augment_sample['documents'][doc_i] = new_doc
            doc_i += 1

        return augment_sample

if __name__ == '__main__':
    # dureader_2.0 / dureader_2.0_v3
    data_version = sys.argv[1]
    source = sys.argv[2]    # search/zhidao

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        new_sample = augment_new_sample(sample, source)
        if new_sample:
            print(json.dumps(new_sample, ensure_ascii=False))
