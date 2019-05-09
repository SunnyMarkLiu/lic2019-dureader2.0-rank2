## prepare
```bash
nohup python run.py --prepare --create_vocab true --data_type search --max_a_len 300 --train_answer_len_cut_bins 6 > search_prepare.log 2>&1 &
nohup python run.py --prepare --create_vocab true --data_type zhidao --max_a_len 400 --train_answer_len_cut_bins 6 > zhidao_prepare.log 2>&1 &
```

## train
```bash
nohup python run.py --train --gpu 0 --data_type search --desc 'pure_v5' --max_a_len 300 --train_answer_len_cut_bins 6 --evaluate_every_batch_cnt 2000 > search_train.log 2>&1 &
nohup python run.py --train --gpu 0 --data_type zhidao --desc 'pure_v5' --max_a_len 400 --train_answer_len_cut_bins 6 --evaluate_every_batch_cnt 2000 > zhidao_train.log 2>&1 &
```

## evaluate
```bash
python run.py --evaluate --gpu 0 --data_type search --desc 'pure_v5' --use_para_prior_scores search --batch_size 128
python run.py --evaluate --gpu 1 --data_type zhidao --desc 'pure_v5' --use_para_prior_scores baidu  --batch_size 128
```

## predict
```bash
nohup python run.py --predict --gpu 0 --data_type search --desc 'pure_v5' --max_a_len 300 --use_para_prior_scores search --batch_size 128 > search_predict.log 2>&1 &
nohup python run.py --predict --gpu 0 --data_type zhidao --desc 'pure_v5' --max_a_len 400 --use_para_prior_scores baidu  --batch_size 128 > zhidao_predict.log 2>&1 &
```
