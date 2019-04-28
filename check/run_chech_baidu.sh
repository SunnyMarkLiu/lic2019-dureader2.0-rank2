#!/usr/bin/env bash

source_extracted_dir="../input/dureader_baidu_preprocess_v0/baidu_preprocess/trainset/"
target_mrc_dir="../input/dureader_baidu_preprocess_v0/mrc_dataset/trainset/"

nohup cat ${source_extracted_dir}split_search_00 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_00 2>&1 &
nohup cat ${source_extracted_dir}split_search_01 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_01 2>&1 &
nohup cat ${source_extracted_dir}split_search_02 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_02 2>&1 &
nohup cat ${source_extracted_dir}split_search_03 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_03 2>&1 &
nohup cat ${source_extracted_dir}split_search_04 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_04 2>&1 &
nohup cat ${source_extracted_dir}split_search_05 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_05 2>&1 &
nohup cat ${source_extracted_dir}split_search_06 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_06 2>&1 &
nohup cat ${source_extracted_dir}split_search_07 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_07 2>&1 &
nohup cat ${source_extracted_dir}split_search_08 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_08 2>&1 &
nohup cat ${source_extracted_dir}split_search_09 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_09 2>&1 &
nohup cat ${source_extracted_dir}split_search_10 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_10 2>&1 &
nohup cat ${source_extracted_dir}split_search_11 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_11 2>&1 &
nohup cat ${source_extracted_dir}split_search_12 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_12 2>&1 &
nohup cat ${source_extracted_dir}split_search_13 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_search_13 2>&1 &

nohup cat ${source_extracted_dir}split_zhidao_00 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_00 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_01 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_01 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_02 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_02 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_03 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_03 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_04 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_04 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_05 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_05 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_06 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_06 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_07 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_07 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_08 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_08 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_09 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_09 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_10 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_10 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_11 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_11 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_12 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_12 2>&1 &
nohup cat ${source_extracted_dir}split_zhidao_13 |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}split_zhidao_13 2>&1 &

source_extracted_dir="../input/dureader_baidu_preprocess_v0/baidu_preprocess/devset/"
target_mrc_dir="../input/dureader_baidu_preprocess_v0/mrc_dataset/devset/"

nohup cat ${source_extracted_dir}search.dev.json |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}search.dev.json 2>&1 &
nohup cat ${source_extracted_dir}zhidao.dev.json |python check_baidu_preprocess_ceil_metric.py > ${target_mrc_dir}zhidao.dev.json 2>&1 &
