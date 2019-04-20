#!/usr/bin/env bash
cd metric

data_version='dureader_2.0'
echo "================== ${data_version} train ceiling results =================="
search_file="../../input/${data_version}/mrc_dataset/devset/search.dev.json"
python mrc_eval.py ${search_file} ${search_file} v1

echo "================== ${data_version} zhidao ceiling results =================="
zhidao_file="../../input/${data_version}/mrc_dataset/trainset/zhidao.dev.json"
python mrc_eval.py ${zhidao_file} ${zhidao_file} v1
