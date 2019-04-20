#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/11 00:43
"""
from models.layers import *
from utils.functions import answer_search


class RNet(torch.nn.Module):
    """
    R-NET model for machine comprehension
    Args:
        - global_config: model_config with types dictionary
    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)
    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, dataset_h5_path):
        super(RNet, self).__init__()

        # set config
        hidden_size = 45
        hidden_mode = 'GRU'
        dropout_p = 0.2
        emb_dropout_p = 0.1
        enable_layer_norm = False

        word_embedding_size = 300
        char_embedding_size = 64
        encoder_word_layers = 3
        encoder_char_layers = 1

        encoder_bidirection = True
        encoder_direction_num = 2 if encoder_bidirection else 1

        match_lstm_bidirection = True
        self_match_lstm_bidirection = True
        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        self_match_rnn_direction_num = 2 if self_match_lstm_bidirection else 1

        ptr_bidirection = True
        self.enable_search = True

        # construct model
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path,
                                            embedding_size=char_embedding_size,
                                            trainable=True)

        self.char_encoder = CharEncoder(mode=hidden_mode,
                                        input_size=char_embedding_size,
                                        hidden_size=hidden_size,
                                        num_layers=encoder_char_layers,
                                        bidirectional=encoder_bidirection,
                                        dropout_p=emb_dropout_p)
        encode_in_size = word_embedding_size + hidden_size * encoder_direction_num

        self.encoder = MyStackedRNN(mode=hidden_mode,
                                    input_size=encode_in_size,
                                    hidden_size=hidden_size,
                                    num_layers=encoder_word_layers,
                                    bidirectional=encoder_bidirection,
                                    dropout_p=emb_dropout_p)
        encode_out_size = hidden_size * encoder_direction_num

        self.match_rnn = MatchRNN(mode=hidden_mode,
                                  hp_input_size=encode_out_size,
                                  hq_input_size=encode_out_size,
                                  hidden_size=hidden_size,
                                  bidirectional=match_lstm_bidirection,
                                  gated_attention=True,
                                  dropout_p=dropout_p,
                                  enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num

        self.self_match_rnn = MatchRNN(mode=hidden_mode,
                                       hp_input_size=match_rnn_out_size,
                                       hq_input_size=match_rnn_out_size,
                                       hidden_size=hidden_size,
                                       bidirectional=self_match_lstm_bidirection,
                                       gated_attention=True,
                                       dropout_p=dropout_p,
                                       enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * self_match_rnn_direction_num

        self.birnn_after_self = MyRNNBase(mode=hidden_mode,
                                          input_size=match_rnn_out_size,
                                          hidden_size=hidden_size,
                                          bidirectional=True,
                                          dropout_p=dropout_p,
                                          enable_layer_norm=enable_layer_norm)
        birnn_out_size = hidden_size * 2

        self.pointer_net = BoundaryPointer(mode=hidden_mode,
                                           input_size=birnn_out_size,
                                           hidden_size=hidden_size,
                                           bidirectional=ptr_bidirection,
                                           dropout_p=dropout_p,
                                           enable_layer_norm=enable_layer_norm)

        self.init_ptr_hidden = AttentionPooling(encode_out_size, hidden_size)

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        assert context_char is not None and question_char is not None

        # word-level embedding: (seq_len, batch, embedding_size)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)

        # char-level embedding: (seq_len, batch, char_embedding_size)
        context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
        question_emb_char, question_char_mask = self.char_embedding.forward(question_char)

        context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
        question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)

        # mix embedding
        context_vec = torch.cat((context_vec, context_vec_char), dim=-1)
        question_vec = torch.cat((question_vec, question_vec_char), dim=-1)

        # encode: (seq_len, batch, hidden_size)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)

        # match lstm: (seq_len, batch, hidden_size)
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_mask,
                                                                               question_encode, question_mask)
        vis_param = {'match': match_para}

        # self match lstm: (seq_len, batch, hidden_size)
        ct_aware_ct, qt_aware_last_hidden, self_para = self.self_match_rnn.forward(qt_aware_ct, context_mask,
                                                                                   qt_aware_ct, context_mask)
        vis_param['self'] = self_para

        # aggregation: (seq_len, batch, hidden_size)
        ct_aware_ct_ag, _ = self.birnn_after_self.forward(ct_aware_ct, context_mask)

        # pointer net init hidden: (batch, hidden_size)
        ptr_net_hidden = self.init_ptr_hidden.forward(question_encode, question_mask)

        # pointer net: (answer_len, batch, context_len)
        ans_range_prop = self.pointer_net.forward(ct_aware_ct_ag, context_mask, ptr_net_hidden)
        ans_range_prop = ans_range_prop.transpose(0, 1)

        # answer range
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)

        return ans_range_prop, ans_range, vis_param
