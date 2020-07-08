# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from convlab2.policy.mle.idea4.GAN.models.generator import LSTMGenerator
from convlab2.policy.mle.idea4.model_dialogue import dialogue_VAE


class SeqGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SeqGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'seqgan'

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)

        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        loss = -torch.sum(pred * reward)

        return loss

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = dialogue_VAE(
        embedding_size= 549,
        rnn_type= "gru",
        hidden_size = 512,
        word_dropout=1,
        embedding_dropout=1,
        latent_size = 256,
        num_layers = 1,
        bidirectional = True
        )
    """
    1. Embedding the current dialogue
    2. Take the last hidden states
    3. Make prediction.
    """
    def forward(self, input):
        """
        :param input: list of input
        :return: B*domain*action*slot
        """
        # embedding
        ele = input[0]
        embedding_dia = self.embedding.compress(input = ele)

        pass
