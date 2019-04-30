#!/usr/bin/env bash

data_version="dureader_2.0_v4"

#------------------------ generate train mrc dataset ------------------------
source_dir="../input/${data_version}/extracted/aug_trainset/"
target_dir="../input/${data_version}/mrc_dataset/aug_trainset/"

nohup cat ${source_dir}split_search_00 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_00 2>&1 &
nohup cat ${source_dir}split_search_01 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_01 2>&1 &
nohup cat ${source_dir}split_search_02 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_02 2>&1 &
nohup cat ${source_dir}split_search_03 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_03 2>&1 &
nohup cat ${source_dir}split_search_04 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_04 2>&1 &
nohup cat ${source_dir}split_search_05 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_05 2>&1 &
nohup cat ${source_dir}split_search_06 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_06 2>&1 &
nohup cat ${source_dir}split_search_07 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_07 2>&1 &
nohup cat ${source_dir}split_search_08 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_08 2>&1 &
nohup cat ${source_dir}split_search_09 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_09 2>&1 &
nohup cat ${source_dir}split_search_10 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_10 2>&1 &
nohup cat ${source_dir}split_search_11 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_11 2>&1 &
nohup cat ${source_dir}split_search_12 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_12 2>&1 &
nohup cat ${source_dir}split_search_13 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_search_13 2>&1 &

nohup cat ${source_dir}split_zhidao_00 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_00 2>&1 &
nohup cat ${source_dir}split_zhidao_01 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_01 2>&1 &
nohup cat ${source_dir}split_zhidao_02 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_02 2>&1 &
nohup cat ${source_dir}split_zhidao_03 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_03 2>&1 &
nohup cat ${source_dir}split_zhidao_04 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_04 2>&1 &
nohup cat ${source_dir}split_zhidao_05 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_05 2>&1 &
nohup cat ${source_dir}split_zhidao_06 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_06 2>&1 &
nohup cat ${source_dir}split_zhidao_07 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_07 2>&1 &
nohup cat ${source_dir}split_zhidao_08 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_08 2>&1 &
nohup cat ${source_dir}split_zhidao_09 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_09 2>&1 &
nohup cat ${source_dir}split_zhidao_10 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_10 2>&1 &
nohup cat ${source_dir}split_zhidao_11 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_11 2>&1 &
nohup cat ${source_dir}split_zhidao_12 |python ../preprocess/4.gen_mrc_dataset.py > ${target_dir}split_zhidao_12 2>&1 &
