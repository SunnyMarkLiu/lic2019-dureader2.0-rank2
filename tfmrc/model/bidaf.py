"""
This module implements the reading comprehension models based on:

1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
3. A Multi-answer Multi-task Framework for Real-world Machine Reading Comprehension
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from util.metric import compute_bleu_rouge
from util.metric import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from layers.loss_func import cul_single_ans_loss, cul_weighted_avg_loss, cul_pas_sel_loss
from tqdm import tqdm
from layers.encoder import transformer_encoder_block


class MultiAnsModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # 额外的设置
        self.config = args

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None], name='p')
        self.q = tf.placeholder(tf.int32, [None, None], name='q')
        self.p_length = tf.placeholder(tf.int32, [None], name='p_len')
        self.q_length = tf.placeholder(tf.int32, [None], name='q_len')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prop')

        if self.config.use_multi_ans_loss: # Answer prediction with multi-answer
            self.logger.info('we use multi answer loss!')
            self.start_label = tf.placeholder(tf.int32, [None, None], name='start_label')
            self.end_label = tf.placeholder(tf.int32, [None, None], name='end_label')
            self.match_score = tf.placeholder(tf.float32, [None, None], name='match_score') # shape=[batch, max_ans_num]
            # TODO 这里是否进行归一化, 如何归一化需要进一步讨论
            self.normed_match_score = tf.div_no_nan(self.match_score, tf.reduce_sum(self.match_score, axis=1, keepdims=True)) # 分数归一化
        else: # baseline
            self.logger.info('we use the baseline single answer loss!')
            self.start_label = tf.placeholder(tf.int32, [None], name='start_label')
            self.end_label = tf.placeholder(tf.int32, [None], name='end_label')

        if self.config.use_rough_classify_feature:
            self.logger.info('use rough classify feature!')
            self.q_rough = tf.placeholder(tf.int32, [None, None], name='q_rough_cls')

        if self.config.use_fine_classify_feature:
            self.logger.info('use fine classify feature!')
            self.q_fine = tf.placeholder(tf.int32, [None, None], name='q_fine_cls')

        if self.config.pos_size: #使用POS特征
            self.logger.info('we use the {} dim pos feature!'.format(self.config.pos_size))
            self.p_pos = tf.placeholder(tf.int32, [None, None], name='p_pos') # shape=[batch*p_num,p_len]
            self.q_pos = tf.placeholder(tf.int32, [None, None], name='q_pos') # shape=[batch*p_num,q_len]

        if self.config.use_pos_freq:
            self.logger.info('we use pos freq feature!')
            self.p_freq = tf.placeholder(tf.float32, [None, None], name='p_pos_freq')
            self.q_freq = tf.placeholder(tf.float32, [None, None], name='q_pos_freq')

        if self.config.use_wiq_feature: # 使用Word-in-Question特征
            self.logger.info('we use the Word-in-Question feature!')
            self.p_wiq = tf.placeholder(tf.int32, [None, None], name='p_wiq') # shape=[batch*p_num,p_len]

        if self.config.use_keyword_feature: #使用 keyword特征
            self.logger.info('we use the keyword feature!')
            self.p_keyword = tf.placeholder(tf.int32, [None, None], name='p_keyword') # shape=[batch*p_num,p_len]
            self.q_keyword = tf.placeholder(tf.int32, [None, None], name='q_keyword') # shape=[batch*p_num,q_len]

        if self.config.ps_loss_weight: # 使用Passage selection Loss
            self.logger.info('we use the passage selection loss, weight is {}'.format(self.config.ps_loss_weight))
            self.gold_passage = tf.placeholder(tf.int32, [None], name='gold_passage') # shape=[batch*p_num]


    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            if self.config.use_oov2unk:
                # 将OOV全部映射为了unk词
                oov_end = 1
            else:
                # 训练unknown和OOV词
                oov_end = self.vocab.oov_word_end_idx + 1

            self.trainable_word_mat = tf.get_variable("trainable_word_emb_mat",
                            [oov_end, self.vocab.embed_dim],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(self.vocab.embedding_matrix[:oov_end],
                            dtype=tf.float32),trainable=True)
            self.pretrained_word_mat = tf.get_variable("pretrained_word_emb_mat",
                            [self.vocab.size() - oov_end, self.vocab.embed_dim],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(self.vocab.embedding_matrix[oov_end:],
                            dtype=tf.float32),trainable=False)

            self.logger.warning('we have {} trainable tokens, we will train them in the model!'
                                    .format(oov_end))

            self.word_embeddings = tf.concat([self.trainable_word_mat, self.pretrained_word_mat], axis=0)

            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

            if self.config.use_rough_classify_feature:
                self.q_emb = tf.concat([self.q_emb, tf.one_hot(self.q_rough, 3, axis=2)], axis=-1)

            if self.config.use_fine_classify_feature:
                self.q_emb = tf.concat([self.q_emb, tf.one_hot(self.q_fine, self.config.fine_cls_num, axis=2)], axis=-1)

            if self.config.pos_size:
                self.p_emb = tf.concat([self.p_emb, tf.one_hot(self.p_pos, self.config.pos_size, axis=2)], axis=-1)
                self.q_emb = tf.concat([self.q_emb, tf.one_hot(self.q_pos, self.config.pos_size, axis=2)], axis=-1)

            if self.config.use_pos_freq:
                self.p_emb = tf.concat([self.p_emb, tf.expand_dims(self.p_freq, axis=2)], axis=-1)
                self.q_emb = tf.concat([self.q_emb, tf.expand_dims(self.q_freq, axis=2)], axis=-1)

            if self.config.use_wiq_feature:
                self.p_emb = tf.concat([self.p_emb, tf.one_hot(self.p_wiq, 2, axis=2)], axis=-1)

            if self.config.use_keyword_feature:
                self.p_emb = tf.concat([self.p_emb, tf.one_hot(self.p_keyword, 2, axis=2)], axis=-1)
                self.q_emb = tf.concat([self.q_emb, tf.one_hot(self.q_keyword, 2, axis=2)], axis=-1)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        # with tf.variable_scope('passage_encoding'):
        #     self.sep_p_encodes = transformer_encoder_block(inputs=self.p_emb, max_input_length=self.max_p_len,
        #                                                    num_conv_layer=16, kernel_size=4, num_att_head=4,
        #                                                    reuse=None)
        # with tf.variable_scope('question_encoding'):
        #     self.sep_q_encodes = transformer_encoder_block(inputs=self.q_emb, max_input_length=self.max_q_len,
        #                                                    num_conv_layer=16, kernel_size=4, num_att_head=4,
        #                                                    reuse=None)

        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        """
        self.loss = 0
        if self.config.use_multi_ans_loss:
            self.weighted_avg_loss = cul_weighted_avg_loss(self.start_probs, self.end_probs, self.start_label,
                                                            self.end_label, self.normed_match_score)
            self.loss += self.weighted_avg_loss
        else: # baseline
            self.single_loss = cul_single_ans_loss(self.start_probs, self.end_probs, self.start_label, self.end_label)
            self.loss += self.single_loss

        if self.config.ps_loss_weight > 0:
            self.pas_sel_loss = cul_pas_sel_loss(self.fuse_p_encodes, self.hidden_size, self.gold_passage)
            self.loss += self.config.ps_loss_weight * self.pas_sel_loss

        if self.config.mrt_loss_weight > 0:
            pass

        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _add_extra_data(self, feed_dict, batch):
        if self.config.use_multi_ans_loss:
            feed_dict[self.start_label] = batch['start_ids']
            feed_dict[self.end_label] = batch['end_ids']
            feed_dict[self.match_score] = batch['match_scores'] # shape =[batch, ans_num]
        else:
            # baseline
            start = []
            end = []
            indexes = np.argmax(batch['match_scores'], axis=1)
            for idx, s, e in zip(indexes, batch['start_ids'], batch['end_ids']):
                start.append(s[idx])
                end.append(e[idx])
            feed_dict[self.start_label] = start
            feed_dict[self.end_label] = end

        if self.config.use_rough_classify_feature:
            feed_dict[self.q_rough] = batch['question_rough_cls']

        if self.config.use_fine_classify_feature:
            feed_dict[self.q_fine] = batch['question_fine_cls']

        if self.config.pos_size: #使用POS特征, shape=[batch_size*max_p_num, max_p_len]
            feed_dict[self.p_pos] = batch['pos_passages']
            feed_dict[self.q_pos] = batch['pos_questions']

        if self.config.use_pos_freq:
            feed_dict[self.p_freq] = batch['pos_freq_passages']
            feed_dict[self.q_freq] = batch['pos_freq_questions']

        if self.config.use_wiq_feature:
            feed_dict[self.p_wiq] = batch['wiq_feature'] # shape=[batch*p_num, p_len]

        if self.config.use_keyword_feature:
            feed_dict[self.p_keyword] = batch['keyword_passages']
            feed_dict[self.q_keyword] = batch['keyword_questions']

        if self.config.ps_loss_weight:
            feed_dict[self.gold_passage] = batch['is_selected'] # shape=[batch*p_num]

        return feed_dict

    def _train_epoch(self, total_batch_count, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0

        tqdm_batch_iterator = tqdm(train_batches, total=total_batch_count)
        for bitx, batch in enumerate(tqdm_batch_iterator):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.dropout_keep_prob: dropout_keep_prob}

            feed_dict = self._add_extra_data(feed_dict, batch)

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            description = "train loss: {:.5f}".format(loss)
            tqdm_batch_iterator.set_description(description)

        return 1.0 * total_loss / total_num

    def train_and_evaluate_several_batchly(self, data, epochs, batch_size, evaluate_every_batch_cnt, save_dir, save_prefix,
              dropout_keep_prob=1.0):
        """
        Train the model with data，batch 的粒度评估 dev 性能
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size: train batch size
            evaluate_every_batch_cnt: evaluate every batch count that training processed
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_rouge_l = 0

        processed_batch_cnt = 0     # 记录 train 处理的 batch 数
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            total_batch_count = data.get_data_length('train') // batch_size + int(data.get_data_length('train') % batch_size != 0)
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)

            # training for one epoch
            epoch_sample_num, epoch_total_loss = 0, 0

            tqdm_batch_iterator = tqdm(train_batches, total=total_batch_count)
            for bitx, batch in enumerate(tqdm_batch_iterator):
                feed_dict = {self.p: batch['passage_token_ids'],
                             self.q: batch['question_token_ids'],
                             self.p_length: batch['passage_length'],
                             self.q_length: batch['question_length'],
                             self.dropout_keep_prob: dropout_keep_prob}
                feed_dict = self._add_extra_data(feed_dict, batch)

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
                epoch_total_loss += loss * len(batch['raw_data'])
                epoch_sample_num += len(batch['raw_data'])

                description = "train loss: {:.5f}".format(loss)
                tqdm_batch_iterator.set_description(description)

                processed_batch_cnt += 1

                # 每处理 evaluate_every_batch_cnt 数的 batch，进行评估
                if evaluate_every_batch_cnt > 0 and processed_batch_cnt % evaluate_every_batch_cnt == 0:
                    self.logger.info('Evaluating the model after processed {} batches'.format(processed_batch_cnt))
                    if data.dev_set is not None:
                        eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                        total_batch_count = data.get_data_length('dev') // batch_size + int(
                            data.get_data_length('dev') % batch_size != 0)
                        eval_loss, bleu_rouge = self.evaluate(total_batch_count, eval_batches)
                        self.logger.info('Dev eval loss {}'.format(eval_loss))
                        self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                        if bleu_rouge['Rouge-L'] > max_rouge_l:
                            self.save(save_dir, save_prefix)
                            max_rouge_l = bleu_rouge['Rouge-L']
                    else:
                        self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, epoch_total_loss / epoch_sample_num))

            # 一个 epoch 之后再进行评估
            self.logger.info('Evaluating the model after epoch {}'.format(epoch))
            if data.dev_set is not None:
                eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                total_batch_count = data.get_data_length('dev') // batch_size + int(data.get_data_length('dev') % batch_size != 0)
                eval_loss, bleu_rouge = self.evaluate(total_batch_count, eval_batches)
                self.logger.info('Dev eval loss {}'.format(eval_loss))
                self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                if bleu_rouge['Rouge-L'] > max_rouge_l:
                    self.save(save_dir, save_prefix)
                    max_rouge_l = bleu_rouge['Rouge-L']
            else:
                self.logger.warning('No dev set is loaded for evaluation in the dataset!')


    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_rouge_l = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            total_batch_count = data.get_data_length('train') // batch_size + int(data.get_data_length('train') % batch_size != 0)
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(total_batch_count, train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    total_batch_count = data.get_data_length('dev') // batch_size + int(data.get_data_length('dev') % batch_size != 0)
                    eval_loss, bleu_rouge = self.evaluate(total_batch_count, eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Rouge-L'] > max_rouge_l:
                        self.save(save_dir, save_prefix)
                        max_rouge_l = bleu_rouge['Rouge-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, total_batch_count, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            total_batch_count: total batch counts
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0

        score_mode = self.config.use_para_prior_scores
        if score_mode == 'baidu':
            pp_scores = (0.44, 0.23, 0.15, 0.09, 0.07)
        elif score_mode == 'zhidao':
            pp_scores = (0.40, 0.22, 0.16, 0.12, 0.10)
        elif score_mode == 'search':
            pp_scores = (0.46, 0.24, 0.15, 0.08, 0.07)
        elif score_mode == 'all':
            pp_scores = (0.43, 0.23, 0.16, 0.10, 0.09)
        elif score_mode == 'best':
            pp_scores = (0.9, 0.05, 0.01, 0.0001, 0.0001)
        else:
            pp_scores = None
        self.logger.info('we use {} model: {} pp_scores'.format(score_mode, pp_scores))

        tqdm_batch_iterator = tqdm(eval_batches, total=total_batch_count)
        for b_itx, batch in enumerate(tqdm_batch_iterator):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.dropout_keep_prob: 1.0}

            feed_dict = self._add_extra_data(feed_dict, batch)

            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer, segmented_pred = self.find_best_answer(sample, start_prob, end_prob,
                                                    padded_p_len, para_prior_scores=pp_scores)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': [],
                                         'segmented_question': sample['segmented_question'],
                                         'segmented_answers': segmented_pred})
                if 'segmented_answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [''.join(seg_ans) for seg_ans in sample['segmented_answers']],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len, para_prior_scores=None):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['documents']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['segmented_passage']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if para_prior_scores is not None:
                # the Nth prior score = the Number of training samples whose gold answer comes
                #  from the Nth paragraph / the number of the training samples
                score *= para_prior_scores[p_idx]
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
            segmented_pred = []
        else:
            segmented_pred = sample['documents'][best_p_idx]['segmented_passage'][best_span[0]: best_span[1] + 1]
            best_answer = ''.join(
                sample['documents'][best_p_idx]['segmented_passage'][best_span[0]: best_span[1] + 1])
        return best_answer, segmented_pred

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
