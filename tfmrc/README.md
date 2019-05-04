## prepare
```bash
nohup python run.py --prepare --create_vocab true --data_type search > search_prepare.log 2>&1 &
nohup python run.py --prepare --create_vocab true --data_type zhidao > zhidao_prepare.log 2>&1 &
```

## train
```bash
nohup python run.py --train --gpu 3 --data_type zhidao --desc 'pure_v5' > zhidao_train.log 2>&1 &
nohup python run.py --train --gpu 3 --data_type search --desc 'pure_v5' > search_train.log 2>&1 &
```

## predict
