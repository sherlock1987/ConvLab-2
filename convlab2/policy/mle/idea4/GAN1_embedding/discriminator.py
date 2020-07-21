#!/usr/bin/env Python
# coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import numpy as np
import collections
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # VAE part
        self.input_hidden_size = 1024
        self.d_layer_embedding = nn.Sequential(nn.Linear(self.input_hidden_size, self.input_hidden_size//4),
                               nn.LeakyReLU(),
                                     nn.Linear(self.input_hidden_size//4, self.input_hidden_size // 12),
                               nn.LeakyReLU(),
                                     nn.Linear(self.input_hidden_size//12, 1),
                               nn.LeakyReLU())

        self.input_hidden_MLP = 549
        self.d_layer_MLP = nn.Sequential(nn.Linear(self.input_hidden_MLP, self.input_hidden_MLP//4),
                               nn.LeakyReLU(),
                                         nn.Linear(self.input_hidden_MLP//4, self.input_hidden_MLP//8),
                               nn.LeakyReLU(),
                                         nn.Linear(self.input_hidden_MLP//8, 1))

        self.output_layer = nn.Sigmoid()

    def forward(self, input_hidden):
        # prob = self.d_layer_MLP(input_hidden)
        prob = self.d_layer_embedding(input_hidden)
        output = self.output_layer(prob)
        return output

    def get_reward(self, r, s, a, mask):
        """
        :param r: reward, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :param s: state, Tensor, [b,340]
        :param a: action, Tensor, [b,209]
        """
        reward_predict = []
        batchsz = r.shape[0]
        s_temp = tensor([]).to(DEVICE)
        a_temp = tensor([]).to(DEVICE)
        reward_collc = []
        """
        # store the data for updating
        data_collc = collections.defaultdict(list)
        data_sub = []
        data_reward_collc = collections.defaultdict(list)    
        """
        for i in range(batchsz):
            # current　states and actions
            s_1 = s[i].unsqueeze(0)
            a_1 = a[i].unsqueeze(0)
            try:
                s_temp = torch.cat((s_temp, s_1), 0)
                a_temp = torch.cat((a_temp, a_1), 0)
            except Exception:
                s_temp = s_1
                a_temp = a_1

            s_train = s_temp.unsqueeze(0).float()
            a_train = a_temp.unsqueeze(0).float()
            input = torch.cat((s_train, a_train), 2)
            if int(mask[i]) == 0:
                # for the last one, the reward should follow the system. 5, 40, -1, that's it.
                last_reward = r[i].item()
                reward_collc = self.get_score_d(input)
                reward_collc[-1] += (last_reward+1)
                # go to bell man
                # bell_man = self.bellman_equ(reward_collc)
                reward_predict += reward_collc
                # clear
                s_temp = tensor([])
                a_temp = tensor([])
                reward_collc = []
            else:
                # process the whole stuff.
                pass

        reward_predict = tensor(reward_predict).to(DEVICE)
        return reward_predict

    def get_score_d(self, input):
        """
        :param input: [1, D, 549] D:dia len
        :return: [ r1, r2, r3,...rD ]
        """

        score = self.forward(input.to(DEVICE)).squeeze(0).squeeze(-1)
        score = self.score_filter(score, minimum=0.001)
        reward_score = (torch.log(score) - 1).tolist()
        return reward_score

    def get_socre_g(self, input):
        """
        from generator to get the reward function.
        :param input: [1, D, 549] D:dia len
        :return: [ r1, r2, r3,...rD ]
        """

        score = self.forward(input).squeeze(0).squeeze(-1)
        score = self.score_filter(score)
        reward_score = (torch.log(score) - 1).tolist()
        return reward_score

    def score_filter(self, score, minimum = 0.00001):
        """
        :param score:
        :return: after filter
        """
        for i in range(score.size(0)):
            tiny = score[i]
            if tiny < minimum:
                score[i] = minimum
        return score

    # def batchClassify(self, inp):
    #     """
    #     Classifies a batch of sequences.
    #
    #     Inputs: inp
    #         - inp: batch_size x seq_len
    #
    #     Returns: out
    #         - out: batch_size ([0,1] score)
    #     """
    #
    #     h = self.init_hidden(inp.size()[0])
    #     out = self.forward(inp, h)
    #     return out.view(-1)
    #
    # def batchBCELoss(self, inp, target):
    #     """
    #     Returns Binary Cross Entropy Loss for discriminator.
    #
    #      Inputs: inp, target
    #         - inp: batch_size x seq_len
    #         - target: batch_size (binary 1/0)
    #     """
    #
    #     loss_fn = nn.BCELoss()
    #     h = self.init_hidden(inp.size()[0])
    #     out = self.forward(inp, h)
    #     return loss_fn(out, target)

