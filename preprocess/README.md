# Preprocess
1. fetch urls and build url mapping dict.
2. text cleaning and word segment, generate POS and Keywords
```bash
split -d --lines 10000 search.train.json split_search_
split -d --lines 10000 zhidao.train.json split_zhidao_
```
3. extract paragraph
4. generate trainable mrc dataset
