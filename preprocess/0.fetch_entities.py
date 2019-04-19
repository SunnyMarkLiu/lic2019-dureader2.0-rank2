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


print('load embeddings dict')
embed_dict = load_pretrained_embed('/home/lq/projects/pretrained_embeddings/chinese/Tencent_AILab_ChineseEmbedding.txt')

all_entities = set()

for raw_f in ['../input/dureader_2.0_v2/raw/trainset/search.train.json',
              '../input/dureader_2.0_v2/raw/trainset/zhidao.train.json',
              '../input/dureader_2.0_v2/raw/devset/search.dev.json',
              '../input/dureader_2.0_v2/raw/devset/zhidao.dev.json',
              '../input/dureader_2.0_v2/raw/devset/cleaned_18.search.dev.json',
              '../input/dureader_2.0_v2/raw/devset/cleaned_18.zhidao.dev.json',
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

with open('./all_entities_dict.txt', 'a', encoding='utf-8') as f:
    f.writelines([entity + '\n' for entity in all_entities])
