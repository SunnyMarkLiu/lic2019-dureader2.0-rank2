#!/usr/bin/env bash

data_version="dureader_2.0_v3"

#------------------------ generate augment train mrc dataset ------------------------
source_mrc_dir="../input/${data_version}/mrc_dataset/trainset/"
target_mrc_dir="../input/${data_version}/extracted/aug_trainset/"

nohup cat ${source_mrc_dir}/split_search_00 |python augment_extracted_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_00 2>&1 &
nohup cat ${source_mrc_dir}/split_search_01 |python augment_extracted_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_01 2>&1 &
nohup cat ${source_mrc_dir}/split_search_02 |python augment_extracted_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_02 2>&1 &
nohup cat ${source_mrc_dir}/split_search_03 |python augment_extracted_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_03 2>&1 &
nohup cat ${source_mrc_dir}/split_search_04 |python augment_extracted_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_04 2>&1 &
nohup cat ${source_mrc_dir}/split_search_05 |python augment_extracted_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_05 2>&1 &
nohup cat ${source_mrc_dir}/split_search_06 |python augment_extracted_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_06 2>&1 &

nohup cat ${source_mrc_dir}/split_zhidao_00 |python augment_extracted_trainset.py dureader_2.0_v3 zhidao > ${target_mrc_dir}/split_zhidao_00 2>&1 &
nohup cat ${source_mrc_dir}/split_zhidao_01 |python augment_extracted_trainset.py dureader_2.0_v3 zhidao > ${target_mrc_dir}/split_zhidao_01 2>&1 &
nohup cat ${source_mrc_dir}/split_zhidao_02 |python augment_extracted_trainset.py dureader_2.0_v3 zhidao > ${target_mrc_dir}/split_zhidao_02 2>&1 &
nohup cat ${source_mrc_dir}/split_zhidao_03 |python augment_extracted_trainset.py dureader_2.0_v3 zhidao > ${target_mrc_dir}/split_zhidao_03 2>&1 &
nohup cat ${source_mrc_dir}/split_zhidao_04 |python augment_extracted_trainset.py dureader_2.0_v3 zhidao > ${target_mrc_dir}/split_zhidao_04 2>&1 &
nohup cat ${source_mrc_dir}/split_zhidao_05 |python augment_extracted_trainset.py dureader_2.0_v3 zhidao > ${target_mrc_dir}/split_zhidao_05 2>&1 &
nohup cat ${source_mrc_dir}/split_zhidao_06 |python augment_extracted_trainset.py dureader_2.0_v3 zhidao > ${target_mrc_dir}/split_zhidao_06 2>&1 &
