#!/usr/bin/env bash

data_version='dureader_2.0_v4'

#------------------------ train raw data ------------------------
source_raw_dir="../input/${data_version}/raw/trainset/"
target_cleaned_dir="../input/${data_version}/cleaned/trainset/"

nohup cat ${source_raw_dir}split_search_00 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_00 2>&1 &
nohup cat ${source_raw_dir}split_search_01 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_01 2>&1 &
nohup cat ${source_raw_dir}split_search_02 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_02 2>&1 &
nohup cat ${source_raw_dir}split_search_03 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_03 2>&1 &
nohup cat ${source_raw_dir}split_search_04 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_04 2>&1 &
nohup cat ${source_raw_dir}split_search_05 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_05 2>&1 &
nohup cat ${source_raw_dir}split_search_06 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_06 2>&1 &
nohup cat ${source_raw_dir}split_search_07 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_07 2>&1 &
nohup cat ${source_raw_dir}split_search_08 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_08 2>&1 &
nohup cat ${source_raw_dir}split_search_09 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_09 2>&1 &
nohup cat ${source_raw_dir}split_search_10 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_10 2>&1 &
nohup cat ${source_raw_dir}split_search_11 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_11 2>&1 &
nohup cat ${source_raw_dir}split_search_12 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_12 2>&1 &
nohup cat ${source_raw_dir}split_search_13 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search_13 2>&1 &

nohup cat ${source_raw_dir}split_zhidao_00 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_00 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_01 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_01 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_02 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_02 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_03 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_03 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_04 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_04 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_05 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_05 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_06 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_06 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_07 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_07 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_08 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_08 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_09 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_09 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_10 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_10 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_11 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_11 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_12 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_12 2>&1 &
nohup cat ${source_raw_dir}split_zhidao_13 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao_13 2>&1 &

#------------------------ dev raw data ------------------------
raw_devset_dir="../input/${data_version}/raw/devset/"
cleaned_devset_dir="../input/${data_version}/cleaned/devset/"

nohup cat ${raw_devset_dir}search.dev.json |python 1.text_cleaning.py > ${cleaned_devset_dir}search.dev.json 2>&1 &
nohup cat ${raw_devset_dir}zhidao.dev.json |python 1.text_cleaning.py > ${cleaned_devset_dir}zhidao.dev.json 2>&1 &

# 去年的 dev 数据
nohup cat ${raw_devset_dir}cleaned_18.search.dev.json |python 1.text_cleaning.py > ${cleaned_devset_dir}cleaned_18.search.dev.json 2>&1 &
nohup cat ${raw_devset_dir}cleaned_18.zhidao.dev.json |python 1.text_cleaning.py > ${cleaned_devset_dir}cleaned_18.zhidao.dev.json 2>&1 &

#------------------------ test raw data ------------------------
source_raw_dir="../input/${data_version}/raw/testset/"
target_cleaned_dir="../input/${data_version}/cleaned/testset/"

nohup cat ${source_raw_dir}split_search1_00 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search1_00 2>&1 &
nohup cat ${source_raw_dir}split_search1_01 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search1_01 2>&1 &
nohup cat ${source_raw_dir}split_search1_02 |python 1.text_cleaning.py > ${target_cleaned_dir}split_search1_02 2>&1 &

nohup cat ${source_raw_dir}split_zhidao1_00 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao1_00 2>&1 &
nohup cat ${source_raw_dir}split_zhidao1_01 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao1_01 2>&1 &
nohup cat ${source_raw_dir}split_zhidao1_02 |python 1.text_cleaning.py > ${target_cleaned_dir}split_zhidao1_02 2>&1 &
