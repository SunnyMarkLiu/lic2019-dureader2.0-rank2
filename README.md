nohup python run.py --train --gpu 3 --desc 'full_data_v5' --train_answer_len_cut_bins 6 \
                    --max_a_len 400 --use_oov2unk false --trainable_oov_cnt_threshold 300 \
                    --rnn_dropout_keep_prob 1 --fuse_dropout_keep_prob 1 --weight_decay 0.00003 \
                    --algo RNET --hidden_size 168 --epochs 20 --evaluate_cnt_in_one_epoch 2 --batch_size 32 \
                    > full_data.train.log 2>&1 &