#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/26 21:50
"""
import torch


class RNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout, only one layer
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm=False):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=1,
                                        bidirectional=bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=1,
                                       bidirectional=bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)

        direction = 2 if bidirectional else 1
        self.out_feature_size = hidden_size * direction
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, seq_batch, mask, device='cpu', final_state=False):
        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = seq_batch.shape
            seq_batch = seq_batch.view(-1, input_size)
            seq_batch = self.layer_norm(seq_batch)
            seq_batch = seq_batch.view(seq_len, batch, input_size)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        # 将 length 为 0 的 id 单独拿出来，非0的参与 rnn 的计算，0 的后续直接拼接对应维度的 zero 向量
        empty_seq_idx = (lengths == 0).nonzero()

        if empty_seq_idx.shape[0] != 0:   # 存在空的seq
            empty_seq_idx = empty_seq_idx.view(empty_seq_idx.shape[0])

            not_empty_seq_idx = (lengths != 0).nonzero().to(device)
            not_empty_seq_idx = not_empty_seq_idx.view(not_empty_seq_idx.shape[0])

            not_empty_lengths = lengths.index_select(0, not_empty_seq_idx)
            not_empty_seq = seq_batch.index_select(1, not_empty_seq_idx)
        else:
            not_empty_lengths = lengths
            not_empty_seq = seq_batch

        not_empty_lengths_sort, not_empty_idx_sort = torch.sort(not_empty_lengths, dim=0, descending=True)
        _, not_empty_idx_unsort = torch.sort(not_empty_idx_sort, dim=0)

        v_sort = not_empty_seq.index_select(1, not_empty_idx_sort)

        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, not_empty_lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # unsorted not empty o
        out_represent = o.index_select(1, not_empty_idx_unsort)  # Note that here first dim is seq_len

        # concate empty input
        if empty_seq_idx.shape[0] != 0:  # 存在空的seq
            out_empty = torch.zeros(seq_batch.shape[0],
                                    seq_batch.shape[1] - not_empty_seq.shape[1],
                                    self.out_feature_size).to(device)
            out_represent = torch.cat([out_represent, out_empty], dim=1)

        # get the last time state
        if final_state:
            len_dix = (lengths - 1).view(-1, 1).expand(-1, out_represent.size(2)).unsqueeze(0)
            state = out_represent.gather(0, len_dix)
            state = state.squeeze(0)
            return out_represent, state
        else:
            return out_represent