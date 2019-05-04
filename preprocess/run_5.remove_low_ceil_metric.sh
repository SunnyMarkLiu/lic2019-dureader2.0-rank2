#!/usr/bin/env bash

data_version="dureader_2.0_v5"

source_dir="../input/${data_version}/mrc_dataset/trainset/"
target_dir="../input/${data_version}/mrc_dataset/final_trainset/"

#------------------------ remove train low ceil rouge-l samples ------------------------
search_min_ceil_rouge=0
search_min_ceil_bleu4=0
zhidao_min_ceil_rouge=0
zhidao_min_ceil_bleu4=0

nohup cat ${source_dir}search.train.json |python 5.remove_low_ceil_metric.py ${search_min_ceil_rouge} ${search_min_ceil_bleu4} > ${target_dir}search.train.json 2>&1 &
nohup cat ${source_dir}zhidao.train.json |python 5.remove_low_ceil_metric.py ${zhidao_min_ceil_rouge} ${zhidao_min_ceil_bleu4} > ${target_dir}zhidao.train.json 2>&1 &

##----------------------- remove aug-train low ceil rouge-l samples ----------------------
#source_dir="../input/${data_version}/mrc_dataset/aug_trainset/"
#target_dir="../input/${data_version}/mrc_dataset/final_trainset/"
#
#search_min_ceil_rouge=50
#search_min_ceil_bleu4=50
#zhidao_min_ceil_rouge=70
#zhidao_min_ceil_bleu4=70
#
#nohup cat ${source_dir}search.train.json |python 5.remove_low_ceil_metric.py ${search_min_ceil_rouge} ${search_min_ceil_bleu4} > ${target_dir}aug_search.train.json 2>&1 &
#nohup cat ${source_dir}zhidao.train.json |python 5.remove_low_ceil_metric.py ${zhidao_min_ceil_rouge} ${zhidao_min_ceil_bleu4} > ${target_dir}aug_zhidao.train.json 2>&1 &
