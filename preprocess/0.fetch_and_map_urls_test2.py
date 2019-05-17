#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
获取 url 链接，作为整体参与分词

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/7 15:29
"""
import sys
import re
import json
import pandas as pd


URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
url_pattern1 = re.compile(URL_REGEX)
URL_REGEX = r'[^\u4e00-\u9fa5|[$-_@.&+]]*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
url_pattern2 = re.compile(URL_REGEX)
all_urls = []

def _fetch_urls(text):
    """ 获取 url 链接，作为整体参与分词 """
    text = text.replace('http：//', 'http://')
    text = text.replace('https：//', 'https://')

    valid_urls = []
    coarse_urls = url_pattern1.findall(text)
    for c_url in coarse_urls:
        c_urls = [c_url]
        if c_url.count('http') > 1:
            c_urls = ['http://' + u for u in c_url.split('http://') if u != '']

        for c_url in c_urls:
            fine_urls = url_pattern2.findall(c_url)
            for f_u in fine_urls:
                if '.' in f_u and len(f_u) > 5:
                    f_u = re.sub('[，。？、￥（）；：…]+', '', f_u)
                    valid_urls.append(f_u)

    if len(valid_urls) > 0:
        all_urls.extend(valid_urls)


# dureader_2.0 / dureader_2.0_v3
data_version = 'dureader_2.0_v5'

for raw_f in [f'../input/{data_version}/raw/test2set/search.test2.json',
              f'../input/{data_version}/raw/test2set/zhidao.test2.json']:
    print('* process', raw_f)
    with open(raw_f, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            documents = sample['documents']
            for document in documents:
                paragraphs = document['paragraphs']
                text = ''.join(paragraphs)
                _fetch_urls(text)

all_urls = list(set(all_urls))

url_map_df = pd.DataFrame()
url_map_df['url'] = all_urls
url_map_df['url_map_id'] = ['url_{}'.format(i) for i in range(len(all_urls))]
url_map_df.to_csv(f'../input/{data_version}/url_mapping_test2.csv', index=False, encoding='utf_8_sig')

# url_xxx 存入 jieba 的自定义词典中
with open(f'../input/{data_version}/all_url_dict_test2.txt', 'w', encoding='utf-8') as f:
    f.writelines([url_id + '\n' for url_id in url_map_df['url_map_id']])
