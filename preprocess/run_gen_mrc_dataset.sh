#!/usr/bin/env bash

data_version="dureader_2.0_v2"

#------------------------ generate train mrc dataset ------------------------
source_extracted_dir="../input/${data_version}/extracted/trainset/"
target_mrc_dir="../input/${data_version}/mrc_dataset/trainset/"

#nohup cat ${source_extracted_dir}split_search_00 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search_00 2>&1 &
#nohup cat ${source_extracted_dir}split_search_01 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search_01 2>&1 &
#nohup cat ${source_extracted_dir}split_search_02 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search_02 2>&1 &
#nohup cat ${source_extracted_dir}split_search_03 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search_03 2>&1 &
#nohup cat ${source_extracted_dir}split_search_04 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search_04 2>&1 &
#nohup cat ${source_extracted_dir}split_search_05 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search_05 2>&1 &
#nohup cat ${source_extracted_dir}split_search_06 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search_06 2>&1 &
#
#nohup cat ${source_extracted_dir}split_zhidao_00 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao_00 2>&1 &
#nohup cat ${source_extracted_dir}split_zhidao_01 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao_01 2>&1 &
#nohup cat ${source_extracted_dir}split_zhidao_02 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao_02 2>&1 &
#nohup cat ${source_extracted_dir}split_zhidao_03 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao_03 2>&1 &
#nohup cat ${source_extracted_dir}split_zhidao_04 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao_04 2>&1 &
#nohup cat ${source_extracted_dir}split_zhidao_05 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao_05 2>&1 &
#nohup cat ${source_extracted_dir}split_zhidao_06 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao_06 2>&1 &

##------------------------ generate dev mrc dataset ------------------------
extracted_devset_dir="../input/${data_version}/extracted/devset/"
mrc_devset_dir="../input/${data_version}/mrc_dataset/devset/"

nohup cat ${extracted_devset_dir}search.dev.json |python 3.gen_mrc_dataset.py > ${mrc_devset_dir}search.dev.json 2>&1 &
nohup cat ${extracted_devset_dir}zhidao.dev.json |python 3.gen_mrc_dataset.py > ${mrc_devset_dir}zhidao.dev.json 2>&1 &

nohup cat ${extracted_devset_dir}cleaned_18.search.dev.json |python 3.gen_mrc_dataset.py > ${mrc_devset_dir}cleaned_18.search.dev.json 2>&1 &
nohup cat ${extracted_devset_dir}cleaned_18.zhidao.dev.json |python 3.gen_mrc_dataset.py > ${mrc_devset_dir}cleaned_18.zhidao.dev.json 2>&1 &
#
##------------------------ generate test mrc dataset ------------------------
#source_extracted_dir="../input/${data_version}/extracted/testset/"
#target_mrc_dir="../input/${data_version}/mrc_dataset/testset/"
#
#nohup cat ${source_extracted_dir}split_search1_00 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search1_00 2>&1 &
#nohup cat ${source_extracted_dir}split_search1_01 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_search1_01 2>&1 &
#
#nohup cat ${source_extracted_dir}split_zhidao1_00 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao1_00 2>&1 &
#nohup cat ${source_extracted_dir}split_zhidao1_01 |python 3.gen_mrc_dataset.py > ${target_mrc_dir}split_zhidao1_01 2>&1 &
