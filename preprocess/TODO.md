# TODO
## 1.text_cleaning.py
- 腾讯词向量的词典用于结巴分词提高覆盖率

## 2.extract_paragraph.py
- 匹配得分的计算：mean(f1) * bleu
- 长度较短的 para 进行拼接，再计算匹配得分 (后期)

## 3.gen_mrc_dataset.py
- 标题是否参与检索 (Done)
- 如果答案直接在原文中，则直接定位（Done，有效）
