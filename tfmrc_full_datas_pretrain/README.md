# Search 和 Zhidao 全量数据预训练模型
## prepare
```bash
nohup python run.py --prepare --create_vocab true --max_a_len 400 --use_oov2unk false \
                    --trainable_oov_cnt_threshold 300 --train_answer_len_cut_bins -1 \
                    > full_data_prepare.log 2>&1 &
```

## train
```bash
nohup python run.py --train --gpu 0 --desc 'full_data_v5' --train_answer_len_cut_bins 6 \
                    --max_a_len 400 --use_oov2unk false --trainable_oov_cnt_threshold 300 \
                    --rnn_dropout_keep_prob 0.95 --fuse_dropout_keep_prob 0.95 --weight_decay 0.00003 \
                    --algo RNET --epochs 20 --evaluate_cnt_in_one_epoch 2 --batch_size 32 \
                    > full_data.train.log 2>&1 &
```

## evaluate
```bash
python run.py --evaluate --gpu 0 --desc 'full_data_v5' --max_a_len 400 \
              --use_oov2unk false --trainable_oov_cnt_threshold 300 --batch_size 128
```
