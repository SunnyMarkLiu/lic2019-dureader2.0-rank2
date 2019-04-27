#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/26 20:22
"""
import torch
import torch.nn as nn
from torchmrc.util import compute_mask


class BasicTokenEmbedder(nn.Module):
    """
    basic embedding layer, and compute the mask of padding index
    """
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 embed_matrix=None,
                 embed_trainable=False,
                 embed_bn=False,
                 padding_idx=0):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embed_dim: The dimension of the word embeddings.
            embed_matrix: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            embed_trainable: Boolean value to indicate whether or not the embedding matrix
                be trainable. Default to False.
            embed_bn: Boolean value to indicate whether or not perform BatchNorm after
                embedding layer
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
        """
        super(BasicTokenEmbedder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_matrix = embed_matrix
        self.embed_trainable = embed_trainable
        self.embed_bn = embed_bn
        self.padding_idx = padding_idx

        self.text_field_embedder = nn.Embedding(self.vocab_size,
                                                self.embed_dim,
                                                padding_idx=self.padding_idx,
                                                _weight=self.embed_matrix)
        # whether the embedding matrix get updated in the learning process or not.
        self.text_field_embedder.weight.requires_grad = self.embed_trainable

        if self.embed_bn:
            # norm every feature dimension with batch input
            self.embedding_batch_norm = nn.BatchNorm1d(num_features=self.embed_dim)
        else:
            self.embedding_batch_norm = None

    def forward(self, x):
        # compute mask vector, 1 when not equal to paddind_idx
        masked_x = compute_mask(x, padding_idx=self.padding_idx)

        embed_x = self.text_field_embedder(x)   # batch_size * seq_len * emb_dim

        if self.embedding_batch_norm is not None:
            embed_x = self.embedding_batch_norm(embed_x.transpose(1, 2).contiguous()).transpose(1, 2)

        embed_x = embed_x.transpose(0, 1)   # seq_len * batch_size * emb_dim

        return embed_x, masked_x
