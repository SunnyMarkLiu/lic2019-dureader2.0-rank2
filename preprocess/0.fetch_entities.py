#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/19 16:29
"""
import json
from tqdm import tqdm

def load_pretrained_embed(filepath):
    """
    load pretrained embeddings
    """
    embeddings_index = set()

    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_size, embed_dim = map(int, f.readline().strip().split(" "))

        for _ in tqdm(range(vocab_size)):
            lists = f.readline().rstrip().split(" ")
            word = lists[0]
            embeddings_index.add(word)

    return embeddings_index


data_version = 'dureader_2.0_v3'
embed_file = '/home/lq/projects/pretrained_embeddings/chinese/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'

print('load embeddings dict')
embed_dict = load_pretrained_embed(embed_file)

all_entities = set()

for raw_f in [f'../input/{data_version}/raw/trainset/search.train.json',
              f'../input/{data_version}/raw/trainset/zhidao.train.json',
              f'../input/{data_version}/raw/devset/search.dev.json',
              f'../input/{data_version}/raw/devset/zhidao.dev.json',
              f'../input/{data_version}/raw/devset/cleaned_18.search.dev.json',
              f'../input/{data_version}/raw/devset/cleaned_18.zhidao.dev.json',
              ]:
    print('* process', raw_f)
    with open(raw_f, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            if 'entity_answers' in sample:
                for entity_ans in sample['entity_answers']:
                    if len(entity_ans) > 0:
                        for entity in entity_ans:
                            if entity != '' and entity in embed_dict and entity not in all_entities:
                                all_entities.add(entity)

print(f'found {len(all_entities)} entities')

with open(f'../input/{data_version}/all_entities_dict_baidu_embed.txt', 'a', encoding='utf-8') as f:
    f.writelines([entity + '\n' for entity in all_entities])
