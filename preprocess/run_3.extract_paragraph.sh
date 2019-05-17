#!/usr/bin/env bash

data_version='dureader_2.0_v5'

# ---------- Hyperparameters ----------
MAX_DOC_LEN=500     # Maximum length of document

#------------------------ extract cleaned  train paragraph ------------------------
source_dir="../input/${data_version}/remove_not_related_paras/trainset/"
target_dir="../input/${data_version}/extracted/trainset/"

nohup cat ${source_dir}split_search_00 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_00 2>&1 &
nohup cat ${source_dir}split_search_01 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_01 2>&1 &
nohup cat ${source_dir}split_search_02 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_02 2>&1 &
nohup cat ${source_dir}split_search_03 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_03 2>&1 &
nohup cat ${source_dir}split_search_04 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_04 2>&1 &
nohup cat ${source_dir}split_search_05 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_05 2>&1 &
nohup cat ${source_dir}split_search_06 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_06 2>&1 &
nohup cat ${source_dir}split_search_07 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_07 2>&1 &
nohup cat ${source_dir}split_search_08 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_08 2>&1 &
nohup cat ${source_dir}split_search_09 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_09 2>&1 &
nohup cat ${source_dir}split_search_10 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_10 2>&1 &
nohup cat ${source_dir}split_search_11 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_11 2>&1 &
nohup cat ${source_dir}split_search_12 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_12 2>&1 &
nohup cat ${source_dir}split_search_13 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_13 2>&1 &

nohup cat ${source_dir}split_zhidao_00 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_00 2>&1 &
nohup cat ${source_dir}split_zhidao_01 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_01 2>&1 &
nohup cat ${source_dir}split_zhidao_02 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_02 2>&1 &
nohup cat ${source_dir}split_zhidao_03 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_03 2>&1 &
nohup cat ${source_dir}split_zhidao_04 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_04 2>&1 &
nohup cat ${source_dir}split_zhidao_05 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_05 2>&1 &
nohup cat ${source_dir}split_zhidao_06 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_06 2>&1 &
nohup cat ${source_dir}split_zhidao_07 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_07 2>&1 &
nohup cat ${source_dir}split_zhidao_08 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_08 2>&1 &
nohup cat ${source_dir}split_zhidao_09 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_09 2>&1 &
nohup cat ${source_dir}split_zhidao_10 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_10 2>&1 &
nohup cat ${source_dir}split_zhidao_11 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_11 2>&1 &
nohup cat ${source_dir}split_zhidao_12 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_12 2>&1 &
nohup cat ${source_dir}split_zhidao_13 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_13 2>&1 &

#------------------------ extract cleaned dev paragraph ------------------------
cleaned_devset_dir="../input/${data_version}/remove_not_related_paras/devset/"
extracted_devset_dir="../input/${data_version}/extracted/devset/"

nohup cat ${cleaned_devset_dir}search.dev.json |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${extracted_devset_dir}search.dev.json 2>&1 &
nohup cat ${cleaned_devset_dir}zhidao.dev.json |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${extracted_devset_dir}zhidao.dev.json 2>&1 &

nohup cat ${cleaned_devset_dir}cleaned_18.search.dev.json |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${extracted_devset_dir}cleaned_18.search.dev.json 2>&1 &
nohup cat ${cleaned_devset_dir}cleaned_18.zhidao.dev.json |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${extracted_devset_dir}cleaned_18.zhidao.dev.json 2>&1 &

#------------------------ extract cleaned dev paragraph ------------------------
source_dir="../input/${data_version}/remove_not_related_paras/testset/"
target_dir="../input/${data_version}/extracted/testset/"

nohup cat ${source_dir}split_search1_00 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search1_00 2>&1 &
nohup cat ${source_dir}split_search1_01 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search1_01 2>&1 &
nohup cat ${source_dir}split_search1_02 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search1_02 2>&1 &

nohup cat ${source_dir}split_zhidao1_00 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao1_00 2>&1 &
nohup cat ${source_dir}split_zhidao1_01 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao1_01 2>&1 &
nohup cat ${source_dir}split_zhidao1_02 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao1_02 2>&1 &

#------------------------ extract cleaned test paragraph ------------------------
source_dir="../input/${data_version}/remove_not_related_paras/test2set/"
target_dir="../input/${data_version}/extracted/test2set/"

nohup cat ${source_dir}split_search_00 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_00 2>&1 &
nohup cat ${source_dir}split_search_01 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_01 2>&1 &
nohup cat ${source_dir}split_search_02 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_02 2>&1 &
nohup cat ${source_dir}split_search_03 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_03 2>&1 &
nohup cat ${source_dir}split_search_04 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_search_04 2>&1 &

nohup cat ${source_dir}split_zhidao_00 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_00 2>&1 &
nohup cat ${source_dir}split_zhidao_01 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_01 2>&1 &
nohup cat ${source_dir}split_zhidao_02 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_02 2>&1 &
nohup cat ${source_dir}split_zhidao_03 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_03 2>&1 &
nohup cat ${source_dir}split_zhidao_04 |python 3.extract_paragraph.py ${MAX_DOC_LEN} > ${target_dir}split_zhidao_04 2>&1 &
