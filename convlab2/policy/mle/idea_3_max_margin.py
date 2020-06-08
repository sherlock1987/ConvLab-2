import os
import torch
import logging
import torch.nn as nn
import numpy as np
from convlab2.util.train_util import to_device
import torch.nn as nn
from torch import optim
import torch.tensor as tensor
import zipfile
import sys
import matplotlib.pyplot  as plt
import pickle
"""
Description for this idea, I try to implement two idea, one is max_margin, and another one is two class Descriminator.
The result show, two class descriminator is much better and stable than another one.
"""
class Reward_max_margin(nn.Module):
    def __init__(self, input_size,output_size):
        super(Reward_max_margin, self).__init__()
        # some of the network is useless
        self.cnn_feature_encoded = nn.Linear(input_size, output_size)
        self.encoder = nn.LSTM(output_size, output_size, batch_first=True, bidirectional=False)
        self.encoder_2 = nn.LSTM(output_size, output_size, batch_first=True, bidirectional=False)
        self.cnn_intput = nn.Linear(input_size,output_size)
        self.cnn_output = nn.Linear(output_size, 1)
        self.norm = int(output_size//20)
        self.margin = 1.
        self.scaler = nn.Sigmoid()

        self.MSE = nn.MSELoss()

    def forward(self, input_feature):
        """
        :param input_feature: [ , , ]
        :return:
        """
        # to construct the batch first, then we could compute the loss function for this stuff, simple and easy.
        input_feature = self.cnn_feature_encoded(input_feature)
        _, (last_hidden, last_cell) = self.encoder(input_feature)
        # second Part
        score = self.cnn_output(last_hidden)
        score = self.scaler(score)
        return score



    def compute_reward(self, input_feature):
        """
        :param input_feature: current dialogue as the input.
        :return: score - 0.5
        """
        _, (last_hidden, last_cell) = self.encoder(self.cnn_feature_encoded(input_feature))
        score = self.cnn_output(last_hidden)
        return score.float()

    def loss_plus_lstm(self, input_real, input_fake):
        """
        :param input_real:
        :param input_fake:
        # pick one of them randomly.
        :return: score and success or fail
        """
        random_seed = torch.randint(low=0,high=2,size = [1])
        res = 0.
        if random_seed.item() == 1:
            _, (last_hidden, last_cell) = self.encoder(self.cnn_feature_encoded(input_real))
            score = self.cnn_output(last_hidden)
            target =torch.tensor([1], dtype=torch.float32, requires_grad=True)
            loss = self.MSE(score,target)
        else:
            _, (last_hidden, last_cell) = self.encoder(self.cnn_feature_encoded(input_fake))
            score = self.cnn_output(last_hidden)
            target =torch.tensor([0], dtype=torch.float32, requires_grad=True)
            loss = self.MSE(score,target)
        # compute success or not
        if score.item() > 0.5:
            if random_seed.item() == 1:
                res = 0
            else:
                res = 1
        else:
            if random_seed.item() == 0:
                res = 0
            else:
                res = 1
        # print(score.item(), target.item(), res)
        return loss.float(),res

    def test(self,input):
        with torch.no_grad():
            return self.scaler(input).item()

    def loss(self, input_real, input_fake, action_real, action_fake):
        """
        :param input_real:[b,se_length,504]
        :param input_fake: [b,se_length,504]
        :param action_real: [1, 209]
        :param action_fake: [1, 209]
        :return:
        """
        score_real = self.forward(input_real)
        score_fake = self.forward(input_fake)
        slack_var = torch.mm(action_real,action_fake.T)/self.norm
        slack_var = 0.
        loss = torch.max(torch.zeros_like(score_real),1 - slack_var + (score_fake - score_real))
        # print("-"*10)
        # print(score_real.item(),score_fake.item())
        # print(self.test(score_real) , self.test(score_fake))
        res_1, res_2 = 0., 0.
        if score_fake.item() < 0.5:
            res_1 = 0
        else:
            res_1 = 1
        if score_real.item() > 0.5:
            res_2 = 0
        else:
            res_2 = 1
        # print(score_fake.item(),res_1,score_real.item(),res_2)
        return loss[0][0], res_1, res_2


    def load(self):
        pass

    def train_iteration(self,input_real, input_fake):
        pass
