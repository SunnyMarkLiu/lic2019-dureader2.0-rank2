#!/usr/bin/env bash

# ---------- Hyperparameters ----------
MAX_DOC_LEN=500     # Maximum length of document
# Minimum match score between paragraph and question/(question+answer)
MIN_MATCH_SCORE_THRESHOLD=1e-100

#------------------------ extract cleaned train paragraph ------------------------
source_cleaned_dir='../input/dureader_2.0/cleaned/trainset/'
target_extracted_dir='../input/dureader_2.0/extracted/trainset/'

nohup cat ${source_cleaned_dir}split_search_00 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_00 2>&1 &
nohup cat ${source_cleaned_dir}split_search_01 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_01 2>&1 &
nohup cat ${source_cleaned_dir}split_search_02 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_02 2>&1 &
nohup cat ${source_cleaned_dir}split_search_03 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_03 2>&1 &
nohup cat ${source_cleaned_dir}split_search_04 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_04 2>&1 &
nohup cat ${source_cleaned_dir}split_search_05 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_05 2>&1 &
nohup cat ${source_cleaned_dir}split_search_06 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_06 2>&1 &
nohup cat ${source_cleaned_dir}split_search_07 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_07 2>&1 &
nohup cat ${source_cleaned_dir}split_search_08 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_08 2>&1 &
nohup cat ${source_cleaned_dir}split_search_09 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_09 2>&1 &
nohup cat ${source_cleaned_dir}split_search_10 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_10 2>&1 &
nohup cat ${source_cleaned_dir}split_search_11 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_11 2>&1 &
nohup cat ${source_cleaned_dir}split_search_12 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_12 2>&1 &
nohup cat ${source_cleaned_dir}split_search_13 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search_13 2>&1 &

nohup cat ${source_cleaned_dir}split_zhidao_00 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_00 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_01 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_01 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_02 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_02 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_03 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_03 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_04 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_04 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_05 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_05 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_06 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_06 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_07 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_07 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_08 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_08 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_09 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_09 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_10 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_10 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_11 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_11 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_12 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_12 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao_13 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao_13 2>&1 &

#------------------------ extract cleaned dev paragraph ------------------------
cleaned_devset_dir='../input/dureader_2.0/cleaned/devset/'
extracted_devset_dir='../input/dureader_2.0/extracted/devset/'

nohup cat ${cleaned_devset_dir}search.dev.json |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}search.dev.json 2>&1 &
nohup cat ${cleaned_devset_dir}zhidao.dev.json |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}zhidao.dev.json 2>&1 &

nohup cat ${cleaned_devset_dir}18.search.dev.json |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}18.search.dev.json 2>&1 &
nohup cat ${cleaned_devset_dir}18.zhidao.dev.json |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${extracted_devset_dir}18.zhidao.dev.json 2>&1 &

#------------------------ extract cleaned dev paragraph ------------------------
source_cleaned_dir='../input/dureader_2.0/cleaned/testset/'
target_extracted_dir='../input/dureader_2.0/extracted/testset/'

nohup cat ${source_cleaned_dir}split_search1_00 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search1_00 2>&1 &
nohup cat ${source_cleaned_dir}split_search1_01 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search1_01 2>&1 &
nohup cat ${source_cleaned_dir}split_search1_02 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_search1_02 2>&1 &

nohup cat ${source_cleaned_dir}split_zhidao1_00 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao1_00 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao1_01 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao1_01 2>&1 &
nohup cat ${source_cleaned_dir}split_zhidao1_02 |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD} > ${target_extracted_dir}split_zhidao1_02 2>&1 &
