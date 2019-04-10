#!/usr/bin/env bash

raw_trainset_dir='../input/dureader_2.0/raw/trainset/'
cleaned_trainset_dir='../input/dureader_2.0/cleaned/trainset/'
extracted_trainset_dir='../input/dureader_2.0/extracted/trainset/'
mrc_trainset_dir='../input/dureader_2.0/mrc_dataset/trainset/'

raw_devset_dir='../input/dureader_2.0/raw/devset/'
cleaned_devset_dir='../input/dureader_2.0/cleaned/devset/'
extracted_devset_dir='../input/dureader_2.0/extracted/devset/'
mrc_devset_dir='../input/dureader_2.0/mrc_dataset/devset/'

raw_testset_dir='../input/dureader_2.0/raw/testset/'
cleaned_testset_dir='../input/dureader_2.0/cleaned/testset/'
extracted_testset_dir='../input/dureader_2.0/extracted/testset/'
mrc_testset_dir='../input/dureader_2.0/mrc_dataset/testset/'

# ---------- Hyperparameters ----------
MAX_DOC_LEN=500     # Maximum length of document
# Minimum match score between paragraph and question/(question+answer)
MIN_MATCH_SCORE_THRESHOLD=1e-100

# step 0
echo '----- fetch all urls, create url mapping ------'
python 0.fetch_urls.py

# step 1
echo '---------------- text cleaning ----------------'
echo '* text cleaning for trainset'
cat ${raw_trainset_dir}search.train.json |python 1.text_cleaning.py > ${cleaned_trainset_dir}search.train.json
cat ${raw_trainset_dir}zhidao.train.json |python 1.text_cleaning.py > ${cleaned_trainset_dir}zhidao.train.json

echo '* text cleaning for devset'
cat ${raw_devset_dir}search.dev.json |python 1.text_cleaning.py > ${cleaned_devset_dir}search.dev.json
cat ${raw_devset_dir}zhidao.dev.json |python 1.text_cleaning.py > ${cleaned_devset_dir}zhidao.dev.json

echo '* text cleaning for testset'
cat ${raw_testset_dir}search.test1.json |python 1.text_cleaning.py > ${cleaned_testset_dir}search.test1.json
cat ${raw_testset_dir}zhidao.test1.json |python 1.text_cleaning.py > ${cleaned_testset_dir}zhidao.test1.json

# step 2
echo '-------------- extract paragraph --------------'
echo '* extract for trainset'
cat ${cleaned_trainset_dir}search.train.json |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD}\
        > ${extracted_trainset_dir}search.train.json
cat ${cleaned_trainset_dir}zhidao.train.json |python 2.extract_paragraph.py train ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD}\
        > ${extracted_trainset_dir}zhidao.train.json

echo '* extract for devset'
cat ${cleaned_devset_dir}search.dev.json |python 2.extract_paragraph.py dev ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD}\
        > ${extracted_devset_dir}search.dev.json
cat ${cleaned_devset_dir}zhidao.train.json |python 2.extract_paragraph.py dev ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD}\
        > ${extracted_devset_dir}zhidao.dev.json

echo '* extract for testset'
cat ${cleaned_testset_dir}search.test1.json |python 2.extract_paragraph.py test ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD}\
        > ${extracted_testset_dir}search.test1.json
cat ${cleaned_testset_dir}zhidao.train.json |python 2.extract_paragraph.py test ${MAX_DOC_LEN} ${MIN_MATCH_SCORE_THRESHOLD}\
        > ${extracted_testset_dir}zhidao.test1.json

# step 3
echo '---------- prepare dataset for model ----------'
echo '* prepare for trainset'
cat ${extracted_trainset_dir}search.train.json |python 3.gen_mrc_dataset.py > ${mrc_trainset_dir}search.train.json
cat ${extracted_trainset_dir}zhidao.train.json |python 3.gen_mrc_dataset.py > ${mrc_trainset_dir}zhidao.train.json

echo '* prepare for devset'
cat ${extracted_devset_dir}search.dev.json |python 3.gen_mrc_dataset.py > ${mrc_devset_dir}search.dev.json
cat ${extracted_devset_dir}zhidao.dev.json |python 3.gen_mrc_dataset.py > ${mrc_devset_dir}zhidao.dev.json

echo '* prepare for testset'
cat ${extracted_testset_dir}search.test1.json |python 3.gen_mrc_dataset.py > ${mrc_testset_dir}search.test1.json
cat ${extracted_testset_dir}zhidao.test1.json |python 3.gen_mrc_dataset.py > ${mrc_testset_dir}zhidao.test1.json
