#!/usr/bin/env bash

data_version="dureader_2.0_v3"

#------------------------ generate augment train mrc dataset ------------------------
source_mrc_dir="../input/${data_version}/mrc_dataset/trainset/"
target_mrc_dir="../input/${data_version}/extracted/aug_trainset/"

nohup cat ${source_mrc_dir}/split_search_00 |python augment_trainset.py dureader_2.0_v3 search > ${target_mrc_dir}/split_search_00
