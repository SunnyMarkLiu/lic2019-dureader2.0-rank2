## prepare
```bash
nohup python run.py --prepare --create_vocab true --data_type search > search_prepare.log 2>&1 &
nohup python run.py --prepare --create_vocab true --data_type zhidao > zhidao_prepare.log 2>&1 &
```

## train
```bash
nohup python run.py --train --gpu 2 --data_type search --desc 'pure_v5' > search_train.log 2>&1 &
nohup python run.py --train --gpu 3 --data_type zhidao --desc 'pure_v5' > zhidao_train.log 2>&1 &
```

## evaluate
```bash
python run.py --evaluate --gpu 0 --data_type search --desc 'pure_v5' --use_para_prior_scores search --batch_size 128
python run.py --evaluate --gpu 1 --data_type zhidao --desc 'pure_v5' --use_para_prior_scores baidu  --batch_size 128
```

## predict
```bash
nohup python run.py --predict --gpu 0 --data_type search --desc 'pure_v5' --use_para_prior_scores search --batch_size 128 > search_predict.log 2>&1 &
nohup python run.py --predict --gpu 1 --data_type zhidao --desc 'pure_v5' --use_para_prior_scores baidu  --batch_size 128 > zhidao_predict.log 2>&1 &
```

# demo
```bash
python run.py --prepare --create_vocab true --data_type search --vocab_min_cnt 1 --desc 'pure_v5_merge_sgns_bigram_char300'
python run.py --train --gpu 2 --data_type search --desc 'pure_v5_merge_sgns_bigram_char300' --epochs 2

```
