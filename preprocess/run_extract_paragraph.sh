#!/usr/bin/env bash

data_version='dureader_2.0_v2'

# ---------- Hyperparameters ----------
MAX_DOC_LEN=500     # Maximum length of document
# Minimum match score between paragraph and question/(question+answer)
MIN_MATCH_SCORE_THRESHOLD=8.644721825390134e-23     # 训练集过滤掉 772417

#------------------------ extract cleaned train paragraph ------------------------
source_cleaned_dir="../input/${data_version}/cleaned/trainset/"
target_extracted_dir="../input/${data_version}/extracted/trainset/"

nohup cat ${source_cleaned_dir}split_search_00 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_00 2>&1 &
nohup cat ${source_cleaned_dir}split_search_01 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_01 2>&1 &
nohup cat ${source_cleaned_dir}split_search_02 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_02 2>&1 &
nohup cat ${source_cleaned_dir}split_search_03 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_03 2>&1 &
nohup cat ${source_cleaned_dir}split_search_04 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_04 2>&1 &
nohup cat ${source_cleaned_dir}split_search_05 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_05 2>&1 &
nohup cat ${source_cleaned_dir}split_search_06 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_06 2>&1 &

nohup cat ${source_cleaned_dir}split_zhidao_00 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_00 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_01 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_01 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_02 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_02 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_03 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_03 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_04 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_04 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_05 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_05 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_06 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_06 2>&1 &

#------------------------ extract cleaned dev paragraph ------------------------
cleaned_devset_dir="../input/${data_version}/cleaned/devset/"
extracted_devset_dir="../input/${data_version}/extracted/devset/"

nohup cat ${cleaned_devset_dir}search.dev.json |python 2.extract_paragraph.py dev ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}search.dev.json 2>&1 &
nohup cat ${cleaned_devset_dir}zhidao.dev.json |python 2.extract_paragraph.py dev ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}zhidao.dev.json 2>&1 &

nohup cat ${cleaned_devset_dir}cleaned_18.search.dev.json |python 2.extract_paragraph.py dev ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}cleaned_18.search.dev.json 2>&1 &
nohup cat ${cleaned_devset_dir}cleaned_18.zhidao.dev.json |python 2.extract_paragraph.py dev ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}cleaned_18.zhidao.dev.json 2>&1 &

#------------------------ extract cleaned dev paragraph ------------------------
source_cleaned_dir="../input/${data_version}/cleaned/testset/"
target_extracted_dir="../input/${data_version}/extracted/testset/"

nohup cat ${source_cleaned_dir}split_search1_00 |python 2.extract_paragraph.py test ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search1_00 2>&1 &
nohup cat ${source_cleaned_dir}split_search1_01 |python 2.extract_paragraph.py test ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search1_01 2>&1 &

nohup cat ${source_cleaned_dir}split_zhidao1_00 |python 2.extract_paragraph.py test ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao1_00 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao1_01 |python 2.extract_paragraph.py test ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao1_01 2>&1 &
