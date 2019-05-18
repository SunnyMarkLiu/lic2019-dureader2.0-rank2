## Ensemble
表格

时间	test1分数	search dev rouge	search dev bleu4	zhidao dev rouge	zhidao dev bleu4
5月8号					
5月10号	58.2||54.54	0.479355014	0.463753881	0.531542002	0.539014601
5月13号	58.47||56.26	0.49053855	0.482444988	0.527933295	0.56267777
5月14号	58.76||58.04	0.488576298	0.491398382	0.548229196	0.555784514
5月15号	59.25||57.72	0.496535539	0.497207651	0.541805142	0.551040785
5月16号	58.62||56.02	0.497134533	0.489828754

```bash
python run_ensemble.py --data_type search --mode dev --max_a_len 300 --use_para_prior_scores search
python run_ensemble.py --data_type zhidao --mode dev --max_a_len 400 --use_para_prior_scores zhidao
```

```bash
nohup python run_ensemble.py --data_type search --mode test --max_a_len 300 --use_para_prior_scores search \
                             > test1.search.ensemble.log 2>&1 &

nohup python run_ensemble.py --data_type search --mode test2 --max_a_len 300 --use_para_prior_scores search \
                             > test2.search.ensemble.log 2>&1 &
                             
                             
nohup python run_ensemble.py --data_type zhidao --mode test --max_a_len 400 --use_para_prior_scores zhidao \
                             > test1.zhidao.ensemble.log 2>&1 &

nohup python run_ensemble.py --data_type zhidao --mode test2 --max_a_len 400 --use_para_prior_scores zhidao \
                             > test2.zhidao.ensemble.log 2>&1 &
```
