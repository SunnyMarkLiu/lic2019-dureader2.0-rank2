## prepare
```bash
nohup python run.py --prepare --create_vocab true --data_type search --desc 'pure_v5' --max_a_len 300 > search_prepare.log 2>&1 &
nohup python run.py --prepare --create_vocab true --data_type zhidao --desc 'pure_v5' --max_a_len 400 > zhidao_prepare.log 2>&1 &
```

## train
```bash
nohup python run.py --train --gpu 2 --data_type search --desc 'pure_v5' --max_a_len 300 --evaluate_every_batch_cnt 2000 > search_train.log 2>&1 &
nohup python run.py --train --gpu 3 --data_type zhidao --desc 'pure_v5' --max_a_len 400 --evaluate_every_batch_cnt 2000 > zhidao_train.log 2>&1 &
```

## evaluate
```bash
python run.py --evaluate --gpu 0 --data_type search --desc 'pure_v5' --use_para_prior_scores search --batch_size 128 --max_a_len 300
python run.py --evaluate --gpu 1 --data_type zhidao --desc 'pure_v5' --use_para_prior_scores baidu  --batch_size 128 --max_a_len 400
```

## predict
```bash
nohup python run.py --predict --gpu 0 --data_type search --desc 'pure_v5' --use_para_prior_scores search --batch_size 128 --max_a_len 300 > search_predict.log 2>&1 &
nohup python run.py --predict --gpu 3 --data_type zhidao --desc 'pure_v5' --use_para_prior_scores baidu  --batch_size 128 --max_a_len 400 > zhidao_predict.log 2>&1 &
```
