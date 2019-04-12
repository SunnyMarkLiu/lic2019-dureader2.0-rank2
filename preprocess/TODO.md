# 2.extract_paragraph.py
- 匹配得分的计算：mean(f1) * bleu
- 长度较短的 para 进行拼接，再计算匹配得分
- <spliter> 的去除

# 3.gen_mrc_dataset.py
- 标题不检索
- 如果答案直接在原文中，则直接定位
