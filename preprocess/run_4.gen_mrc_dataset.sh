#!/usr/bin/env bash

data_version="dureader_2.0_v5"

#------------------------ generate train mrc dataset ------------------------
source_dir="../input/${data_version}/extracted/trainset/"
target_dir="../input/${data_version}/mrc_dataset/trainset/"

nohup cat ${source_dir}split_search_00 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_00 2>&1 &
nohup cat ${source_dir}split_search_01 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_01 2>&1 &
nohup cat ${source_dir}split_search_02 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_02 2>&1 &
nohup cat ${source_dir}split_search_03 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_03 2>&1 &
nohup cat ${source_dir}split_search_04 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_04 2>&1 &
nohup cat ${source_dir}split_search_05 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_05 2>&1 &
nohup cat ${source_dir}split_search_06 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_06 2>&1 &
nohup cat ${source_dir}split_search_07 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_07 2>&1 &
nohup cat ${source_dir}split_search_08 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_08 2>&1 &
nohup cat ${source_dir}split_search_09 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_09 2>&1 &
nohup cat ${source_dir}split_search_10 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_10 2>&1 &
nohup cat ${source_dir}split_search_11 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_11 2>&1 &
nohup cat ${source_dir}split_search_12 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_12 2>&1 &
nohup cat ${source_dir}split_search_13 |python 4.gen_mrc_dataset.py > ${target_dir}split_search_13 2>&1 &

nohup cat ${source_dir}split_zhidao_00 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_00 2>&1 &
nohup cat ${source_dir}split_zhidao_01 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_01 2>&1 &
nohup cat ${source_dir}split_zhidao_02 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_02 2>&1 &
nohup cat ${source_dir}split_zhidao_03 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_03 2>&1 &
nohup cat ${source_dir}split_zhidao_04 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_04 2>&1 &
nohup cat ${source_dir}split_zhidao_05 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_05 2>&1 &
nohup cat ${source_dir}split_zhidao_06 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_06 2>&1 &
nohup cat ${source_dir}split_zhidao_07 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_07 2>&1 &
nohup cat ${source_dir}split_zhidao_08 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_08 2>&1 &
nohup cat ${source_dir}split_zhidao_09 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_09 2>&1 &
nohup cat ${source_dir}split_zhidao_10 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_10 2>&1 &
nohup cat ${source_dir}split_zhidao_11 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_11 2>&1 &
nohup cat ${source_dir}split_zhidao_12 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_12 2>&1 &
nohup cat ${source_dir}split_zhidao_13 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao_13 2>&1 &

#------------------------ generate dev mrc dataset ------------------------
source_dir="../input/${data_version}/extracted/devset/"
target_dir="../input/${data_version}/mrc_dataset/devset/"

nohup cat ${source_dir}search.dev.json |python 4.gen_mrc_dataset.py > ${target_dir}search.dev.json 2>&1 &
nohup cat ${source_dir}zhidao.dev.json |python 4.gen_mrc_dataset.py > ${target_dir}zhidao.dev.json 2>&1 &

nohup cat ${source_dir}cleaned_18.search.dev.json |python 4.gen_mrc_dataset.py > ${target_dir}cleaned_18.search.dev.json 2>&1 &
nohup cat ${source_dir}cleaned_18.zhidao.dev.json |python 4.gen_mrc_dataset.py > ${target_dir}cleaned_18.zhidao.dev.json 2>&1 &

#------------------------ generate test mrc dataset ------------------------
source_dir="../input/${data_version}/extracted/testset/"
target_dir="../input/${data_version}/mrc_dataset/testset/"

nohup cat ${source_dir}split_search1_00 |python 4.gen_mrc_dataset.py > ${target_dir}split_search1_00 2>&1 &
nohup cat ${source_dir}split_search1_01 |python 4.gen_mrc_dataset.py > ${target_dir}split_search1_01 2>&1 &
nohup cat ${source_dir}split_search1_02 |python 4.gen_mrc_dataset.py > ${target_dir}split_search1_02 2>&1 &

nohup cat ${source_dir}split_zhidao1_00 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao1_00 2>&1 &
nohup cat ${source_dir}split_zhidao1_01 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao1_01 2>&1 &
nohup cat ${source_dir}split_zhidao1_02 |python 4.gen_mrc_dataset.py > ${target_dir}split_zhidao1_02 2>&1 &
