#!/usr/bin/env bash

# step 1
echo '------- text cleaning -------'
echo '* text cleaning for trainset'
cat ../input/dureader_2.0/raw/trainset/search.train.json |python 1.text_cleaning.py > ../input/dureader_2.0/cleaned/trainset/search.train.json
#cat ../input/dureader_2.0/raw/trainset/zhidao.train.json |python 1.text_cleaning.py > ../input/dureader_2.0/cleaned/trainset/zhidao.train.json
echo '* text cleaning for devset'
cat ../input/dureader_2.0/raw/devset/search.dev.json |python 1.text_cleaning.py > ../input/dureader_2.0/cleaned/devset/search.dev.json
#cat ../input/dureader_2.0/raw/devset/zhidao.dev.json |python 1.text_cleaning.py > ../input/dureader_2.0/cleaned/devset/zhidao.dev.json
echo '* text cleaning for testset'
cat ../input/dureader_2.0/raw/testset/search.test1.json |python 1.text_cleaning.py > ../input/dureader_2.0/cleaned/testset/search.test1.json
#cat ../input/dureader_2.0/raw/testset/zhidao.test1.json |python 1.text_cleaning.py > ../input/dureader_2.0/cleaned/testset/zhidao.test1.json
