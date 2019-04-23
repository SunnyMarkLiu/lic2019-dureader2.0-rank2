# 数据增强策略
针对 V3 的 mrc_dataset/trainset 数据，其中只有一个 gold answer，思路是将gold answer去除，将剩下的
第二好的答案作为新的 gold answer，question + answer 检索出document，注意和原始document的重复率问题。
