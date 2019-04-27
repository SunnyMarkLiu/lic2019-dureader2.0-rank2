#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/26 21:41
"""
import torch
import torch.nn.functional as F


class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
    and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
    it to every time step.
    """

    def forward(self, input_tensor):
        # pylint: disable=arguments-differ
        """
        Apply dropout to input tensor.
        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        # (batch_size, embedding_dim)
        dropout_mask = F.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            # (batch_size, 1， embedding_dim) * (batch_size, num_timesteps, embedding_dim)
            return dropout_mask.unsqueeze(1) * input_tensor
