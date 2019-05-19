# Ensemble
## devset
```bash
python run_ensemble.py --data_type search --mode dev --max_a_len 300 --use_para_prior_scores search
python run_ensemble.py --data_type zhidao --mode dev --max_a_len 400 --use_para_prior_scores zhidao
```

## testset
```bash
nohup python run_ensemble.py --data_type search --mode test --max_a_len 300 --use_para_prior_scores search \
                             > test1.search.ensemble.log 2>&1 &

nohup python run_ensemble.py --data_type zhidao --mode test --max_a_len 400 --use_para_prior_scores zhidao \
                             > test1.zhidao.ensemble.log 2>&1 &


nohup python run_ensemble.py --data_type search --mode test2 --max_a_len 300 --use_para_prior_scores search \
                             > test2.search.ensemble.log 2>&1 &

nohup python run_ensemble.py --data_type zhidao --mode test2 --max_a_len 400 --use_para_prior_scores zhidao \
                             > test2.zhidao.ensemble.log 2>&1 &
```
