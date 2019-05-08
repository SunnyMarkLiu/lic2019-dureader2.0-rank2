#!/usr/bin/env bash

data_version="dureader_2.0_v5"

source_dir="../input/${data_version}/mrc_dataset/final_trainset/"
target_dir="../input/${data_version}/final_mrc_dataset/trainset/"

nohup cat ${source_dir}search.train.json |python 6.generate_para_match_score_feature.py > ${target_dir}search.train.json 2>&1 &
nohup cat ${source_dir}zhidao.train.json |python 6.generate_para_match_score_feature.py > ${target_dir}zhidao.train.json 2>&1 &

source_dir="../input/${data_version}/mrc_dataset/devset/"
target_dir="../input/${data_version}/final_mrc_dataset/devset/"

nohup cat ${source_dir}search.dev.json |python 6.generate_para_match_score_feature.py > ${target_dir}search.dev.json 2>&1 &
nohup cat ${source_dir}zhidao.dev.json |python 6.generate_para_match_score_feature.py > ${target_dir}zhidao.dev.json 2>&1 &
nohup cat ${source_dir}cleaned_18.search.dev.json |python 6.generate_para_match_score_feature.py > ${target_dir}cleaned_18.search.dev.json 2>&1 &
nohup cat ${source_dir}cleaned_18.zhidao.dev.json |python 6.generate_para_match_score_feature.py > ${target_dir}cleaned_18.zhidao.dev.json 2>&1 &

source_dir="../input/${data_version}/mrc_dataset/testset/"
target_dir="../input/${data_version}/final_mrc_dataset/testset/"

nohup cat ${source_dir}search.test1.json |python 6.generate_para_match_score_feature.py > ${target_dir}search.test1.json 2>&1 &
nohup cat ${source_dir}zhidao.test1.json |python 6.generate_para_match_score_feature.py > ${target_dir}zhidao.test1.json 2>&1 &
