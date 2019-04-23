#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/23 10:16
"""
from pprint import pprint
from elasticsearch import Elasticsearch

dureader_index_name = "dureader_data_augment"

TYPE_NAME = "_doc"
ES_HOST = "127.0.0.1"
ES_USER = "sunnymarkliu"
ES_PASSWD = "liuqing;"

print("start to create index: '%s'" % dureader_index_name)

# connect to ES
try:
    es_client = Elasticsearch(hosts=[ES_HOST], http_auth=(ES_USER, ES_PASSWD))
except:
    es_client = Elasticsearch(hosts=[ES_HOST])

pprint(es_client.info())

if es_client.indices.exists(dureader_index_name):
    print("deleting '%s' index..." % dureader_index_name)
    res = es_client.indices.delete(index=dureader_index_name)
    print("response: '%s'" % res)

request_body = {
    "settings": {
        "number_of_shards": 5,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "whitespace_remove": {
                    "type": "pattern_replace",
                    "pattern": " ",
                    "replacement": ""
                },
                "remove_empty": {
                    "type": "length",
                    "min": 1
                },
                "chinese_stop": {
                    "type": "stop",
                    "stopwords_path": "analysis/chinese_stopword.txt"
                },
                "eng_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                },
                "filter_stop": {
                    "type": "stop",
                },
                "filter_shingle": {
                    "type": "shingle",
                    "max_shingle_size": 2,
                    "min_shingle_size": 2,
                    "output_unigrams": "true",
                    "filler_token": ""
                },
                "jieba_stop": {
                    "type": "stop",
                    "stopwords_path": "stopwords/stopword_all.txt"
                },
                "jieba_synonym_alias_search": {
                    "type": "synonym_graph",
                    "synonyms_path": "synonyms/alias.syno"
                },
                "jieba_synonym_all_search": {
                    "type": "synonym_graph",
                    "synonyms_path": "synonyms/all.syno"
                },
                "jieba_synonym_alias_index": {
                    "type": "synonym",
                    "synonyms_path": "synonyms/alias.syno"
                },
                "jieba_synonym_all_index": {
                    "type": "synonym",
                    "synonyms_path": "synonyms/all.syno"
                },
                "syno_filter": {
                    "type": "synonym",
                    "synonyms_path": "analysis/symptom_mapping_alphaMLE.txt"
                },
                "new_syno": {
                    "type": "synonym",
                    "synonyms_path": "analysis/chinese_synonymous_simple.txt"
                },
            },
            "analyzer": {
                "es_ngram_analyzer": {
                    "type": "custom",
                    "tokenizer": "ngram_tokenizer",
                    "filter": [
                        "lowercase",
                        "chinese_stop",
                        "eng_stop"
                    ]
                },
                "ngram_ana": {
                    "filter": [
                        "lowercase",
                        "jieba_stop",
                        "eng_stop"
                    ],
                    "type": "custom",
                    "tokenizer": "ngram_tokenizer"
                },
                "jieba_index_ana": {
                    "tokenizer": "jieba_index",
                    "filter": [
                        "lowercase",
                        "jieba_synonym_all_index",
                        "whitespace_remove",
                        "remove_empty"
                    ]
                },
                "jieba_search_ana": {
                    "tokenizer": "jieba_index",
                    "filter": [
                        "lowercase",
                        "whitespace_remove",
                        "remove_empty"
                    ]
                },
                "analyzer_shingle": {
                    "tokenizer": "ngram",
                    "filter": ["standard", "lowercase", "filter_stop", "filter_shingle"]
                }

            },
            "tokenizer": {
                "ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 2,
                    "max_gram": 3,
                    "token_chars": [
                        "letter",
                        "digit"
                    ]
                }
            }
        }
    },
    "mappings": {
        "_doc": {
            "properties": {
                "segmented_passage": {
                    "type": "text",
                    "analyzer": "jieba_search_ana",
                    "search_analyzer": "jieba_search_ana"
                },
                "pos_passage": {
                    "type": "text"
                },
                "keyword_passage": {
                    "type": "text"
                },
                "passage_word_in_question": {
                    "type": "text"
                },
                "title_len": {
                    "type": "integer"
                },
                "source": {
                    "type": "text"
                }
            }
        }
    }
}

print("creating '%s' index..." % dureader_index_name)
res = es_client.indices.create(index=dureader_index_name, body=request_body, request_timeout=30)
print("response: '%s'" % res)
