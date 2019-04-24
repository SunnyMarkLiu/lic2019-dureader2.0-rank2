#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
将训练集的所有 document 存储到 ES 中

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/23 14:03
"""
import sys
import json
from pprint import pprint
from elasticsearch import Elasticsearch
from elasticsearch import helpers

dureader_index_name = "dureader_data_augment"
TYPE_NAME = "_doc"
ES_HOST = "127.0.0.1"

# predefined splitter
splitter = u'<splitter>'

print('connect to ES server')
es_client = Elasticsearch(hosts=[ES_HOST])

pprint(es_client.info())

# dureader_2.0 / dureader_2.0_v3
data_version = sys.argv[1]

original_train_files = {
    'search': f'../input/{data_version}/mrc_dataset/trainset/search.train.json',
    'zhidao': f'../input/{data_version}/mrc_dataset/trainset/zhidao.train.json'
}

doc_count = es_client.count(index=dureader_index_name)['count']
print('ES already has {} documents'.format(doc_count))

insert_actions = []
for search_zhidao in original_train_files:
    print(f'process {original_train_files[search_zhidao]}...')
    with open(original_train_files[search_zhidao], 'r') as f:
        for line in f:
            if not line.startswith('{'): continue

            sample = json.loads(line.strip())
            for doc in sample['documents']:
                action = {
                    "_index": dureader_index_name,
                    "_id": doc_count,
                    "_type": "_doc",
                    "_source": {
                        'segmented_passage': '<es_splitter>'.join(doc['segmented_passage']),
                        'pos_passage': '<es_splitter>'.join(doc['pos_passage']),
                        'keyword_passage': '<es_splitter>'.join(list(map(str, doc['keyword_passage']))),
                        'passage_word_in_question': '<es_splitter>'.join(list(map(str, doc['passage_word_in_question']))),
                        'title_len': doc['title_len'],
                        'source': search_zhidao
                    }
                }
                insert_actions.append(action)
                doc_count += 1

                if len(insert_actions) % 10000 == 0:
                    print('current doc count {}'.format(doc_count))
                    helpers.bulk(es_client, insert_actions)
                    insert_actions = []

# 插入剩余的 doc
helpers.bulk(es_client, insert_actions)
print('current doc count {}'.format(doc_count))
