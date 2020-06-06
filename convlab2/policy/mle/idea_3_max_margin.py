import os
import torch
import logging
import torch.nn as nn
import numpy as np
from convlab2.util.train_util import to_device
import torch.nn as nn
from torch import optim

import zipfile
import sys
import matplotlib.pyplot  as plt
import pickle

class Reward_max(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Reward_max, self).__init__()
        self.encoder_1 = nn.LSTM(input_size, output_size, batch_first=True, bidirectional=False)
        self.encoder_2 = nn.LSTM(output_size, output_size)

        self.m = nn.Sigmoid()
        self.loss = nn.BCELoss(size_average= False,reduce= True)
        self.cnn_belief = nn.Linear(input_size-output_size,output_size)
        self.cnn_output = nn.Linear(output_size,output_size)

    def forward(self, input_feature, input_belief, target):
        # to construct the batch first, then we could compute the loss function for this stuff, simple and easy.
        _, (last_hidden, last_cell) = self.encoder_1(input_feature)
        # second Part

        _, (predict_action, last_cell) = self.encoder_2(self.cnn_belief(input_belief), (last_hidden, last_cell))

        loss = self.loss(self.m(self.cnn_output(predict_action)),target)
        return loss

    def compute_reward(self, input_feature, input_belief, input_predict_RL):
        # compute the reward based on the very easy methoc
        _, (last_hidden, last_cell) = self.encoder_1(input_feature)
        # second Part
        _, (predict_action, last_cell) = self.encoder_2(self.cnn_belief(input_belief), (last_hidden, last_cell))
        action_prob = self.m(self.cnn_output(predict_action))
        reward = action_prob.unsqueeze(0) * input_predict_RL.unsqueeze(0)
        res = torch.sum(reward.unsqueeze(0).unsqueeze(0))
        return res
    def fake_data_generator(self):
        pass