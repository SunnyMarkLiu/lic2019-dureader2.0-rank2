#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/26 19:20
"""
import torch
import torch.nn as nn
from torchmrc.modules.embedding import BasicTokenEmbedder
from torchmrc.modules.encoder import RNNBase
from torchmrc.modules.match import MatchRNN
from torchmrc.modules.pointer_net import BoundaryPointer
from torchmrc.functions import answer_search


class MatchLSTM(nn.Module):
    """
    Implement MatchLSTM model for machine reading comprehension.
    """

    def __init__(self,
                 max_p_num,
                 max_p_len,
                 max_q_len,

                 vocab_size,
                 embed_dim,
                 rnn_mode,
                 hidden_size,
                 encoder_bidirection,
                 match_lstm_bidirection,
                 gated_attention,
                 rnn_dropout,
                 enable_layer_norm,
                 ptr_bidirection,
                 embed_matrix=None,
                 embed_trainable=False,
                 embed_bn=False,
                 padding_idx=0,
                 embed_dropout=0,
                 device='cpu',
                 enable_search=True):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embed_dim: The dimension of the word embeddings.
            rnn_mode: The type of rnn, LSTM/GRU.
            hidden_size: hidden size of rnn.
            embed_matrix: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            embed_trainable: Boolean value to indicate whether or not the embedding matrix
                be trainable. Default to False.
            embed_bn: Boolean value to indicate whether or not perform BatchNorm after
                embedding layer
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            embed_dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
        """
        super(MatchLSTM, self).__init__()
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_matrix = embed_matrix
        self.embed_trainable = embed_trainable
        self.embed_bn = embed_bn
        self.padding_idx = padding_idx
        self.embed_dropout = embed_dropout

        self.rnn_mode = rnn_mode
        self.hidden_size = hidden_size
        self.encoder_bidirection = encoder_bidirection
        encoder_direction_num = 2 if self.encoder_bidirection else 1

        self.match_lstm_bidirection = match_lstm_bidirection
        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        self.gated_attention = gated_attention
        self.rnn_dropout = rnn_dropout
        self.enable_layer_norm = enable_layer_norm

        self.ptr_bidirection = ptr_bidirection

        self.device = device
        self.enable_search = enable_search

        # ------ level 0: embedding layer ------
        self.text_field_embedder = BasicTokenEmbedder(vocab_size=self.vocab_size,
                                                      embed_dim=self.embed_dim,
                                                      embed_matrix=self.embed_matrix,
                                                      embed_trainable=self.embed_trainable,
                                                      embed_bn=self.embed_bn,
                                                      padding_idx=self.padding_idx)

        # ------ level 1: input context encoding layer ------
        self.encoder = RNNBase(mode=self.rnn_mode,
                               input_size=self.embed_dim,
                               hidden_size=self.hidden_size,
                               bidirectional=self.encoder_bidirection,
                               dropout_p=self.embed_dropout)
        encode_out_size = hidden_size * encoder_direction_num

        # ------ level 2: matching layer ------
        self.match_rnn = MatchRNN(mode=self.rnn_mode,
                                  hp_input_size=encode_out_size,
                                  hq_input_size=encode_out_size,
                                  hidden_size=self.hidden_size,
                                  bidirectional=self.match_lstm_bidirection,
                                  gated_attention=self.gated_attention,
                                  dropout_p=self.rnn_dropout,
                                  enable_layer_norm=self.enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num

        self.pointer_net = BoundaryPointer(mode=self.rnn_mode,
                                           input_size=match_rnn_out_size,
                                           hidden_size=self.hidden_size,
                                           bidirectional=self.ptr_bidirection,
                                           dropout_p=self.rnn_dropout,
                                           enable_layer_norm=self.enable_layer_norm)


    def forward(self, question, context, passage_cnts):
        # get embedding: (seq_len, batch, embedding_size)
        question_vec, question_mask = self.text_field_embedder.forward(question)
        context_vec, context_mask = self.text_field_embedder.forward(context)

        # encode: (seq_len, batch, hidden_size)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)

        # match lstm: (seq_len, batch, hidden_size)
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_mask,
                                                                               question_encode, question_mask)
        # concate passage with same question
        batch_sample_qt_aware_ct = []
        batch_sample_context_mask = []

        start = 0
        for passage_cnt in passage_cnts:
            batch_sample_idx = torch.arange(start=start, end=start+passage_cnt).to(self.device)
            start += passage_cnt

            # split batched passage
            sample_qt_aware_ct = qt_aware_ct.index_select(1, batch_sample_idx)
            sample_qt_aware_ct = sample_qt_aware_ct.view(-1, 1, sample_qt_aware_ct.shape[-1])

            sample_context_mask = context_mask.index_select(0, batch_sample_idx)
            sample_context_mask = sample_context_mask.view(1, passage_cnt * sample_context_mask.shape[-1])
            # padding to max total passage length

            batch_sample_qt_aware_ct.append(sample_qt_aware_ct)
            batch_sample_context_mask.append(sample_context_mask)

        qt_aware_ct = torch.cat(batch_sample_qt_aware_ct, 1)
        context_mask = torch.cat(batch_sample_context_mask, 0)

        vis_param = {'match': match_para}

        # pointer net: (answer_len, batch, context_len)
        ans_range_prop = self.pointer_net.forward(qt_aware_ct, context_mask)
        ans_range_prop = ans_range_prop.transpose(0, 1)     # (batch, answer_len, context_len), answer_len=2

        # answer range
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)

        return ans_range_prop, ans_range, vis_param
