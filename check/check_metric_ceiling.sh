#!/usr/bin/env bash
cd metric
starttime=`date +'%Y-%m-%d %H:%M:%S'`

data_version='dureader_2.0'

echo "================== ${data_version} train ceiling results =================="
echo "search:"
mrc_file="../../input/${data_version}/mrc_dataset/trainset/search.train.json"
python mrc_eval.py ${mrc_file} ${mrc_file} v1
echo "zhidao:"
mrc_file="../../input/${data_version}/mrc_dataset/trainset/zhidao.train.json"
python mrc_eval.py ${mrc_file} ${mrc_file} v1

echo "================== ${data_version} dev ceiling results =================="
echo "search:"
mrc_file="../../input/${data_version}/mrc_dataset/devset/search.dev.json"
python mrc_eval.py ${mrc_file} ${mrc_file} v1
echo "zhidao:"
mrc_file="../../input/${data_version}/mrc_dataset/devset/zhidao.dev.json"
python mrc_eval.py ${mrc_file} ${mrc_file} v1

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "Total cost time: "$((end_seconds-start_seconds))"s"
