#!/usr/bin/env Python
# coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from convlab2.policy.mle.idea4.model_dialogue import dialogue_VAE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

class Generator(nn.Module):
    def __init__(self,  hidden_size, output_size, bf_inp_size = 340, a_inp_size = 209, gpu=False):
        # Todo make this looks a little good.
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gpu = gpu

        self.embeddings = dialogue_VAE(
        embedding_size= 549,
        rnn_type= "gru",
        hidden_size = 512,
        word_dropout=1,
        embedding_dropout=1,
        latent_size = 256,
        num_layers = 1,
        bidirectional = True
        )
        self.input_size = 512*2 + 340
        self.l1 = nn.Linear(self.input_size, 682)
        self.l2 = nn.Linear(682, 512)
        self.l3 = nn.Linear(512, 304)
        self.emb_bf_layer = nn.Sequential(nn.Linear(self.input_size, 1000),
                               nn.ReLU(),
                               nn.Linear(1000, 500),
                               nn.ReLU(),
                               nn.Linear(500,209))

        self.bf_layer = nn.Sequential(nn.Linear(340, 300),
                               nn.ReLU(),
                               nn.Linear(300, 250),
                               nn.ReLU(),
                               nn.Linear(250,209))
        self.bf_layer_right = nn.Sequential(nn.Linear(340, 300),
                               nn.ReLU(),
                               nn.Linear(300, 250),
                               nn.ReLU(),
                               nn.Linear(250,209))
        self.bf_plus = nn.Linear(1,2)
        self.bf_bf = nn.Linear(340, 340)
        self.output_layer = nn.Sigmoid()

    def output_layer_gumbel_softmax(self, input):
        return input

    def load_VAE(self, path):
        abc = (torch.load(path, map_location=DEVICE))
        self.embeddings.load_state_dict(torch.load(path, map_location=DEVICE))

    # should be no use.
    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if self.gpu:
            return h.cuda()
        else:
            return h

    """
    1. Embedding the current dialogue
    2. Take the last hidden states, cat it with bf
    3. Make prediction, sequential classification.
    3.1 First I will try predict 300 actions, next I will try gumbel softmax.
    3.2 Next I will try add embedding. Add embedding is much better.
    """
    def predict_act(self, prev_list, bf):
        """
        :param prev_list:
        :param bf: [1, batch_size, 549]
        :return:
        """
        # hidden = self.forward_embedding(prev_list, bf)
        hidden = self.forward_bf(bf)
        # hidden_prob = self.output_layer(hidden).to("cuda")
        # make action looks like real
        # no gradient
        zero = torch.zeros_like(hidden)
        one = torch.ones_like(hidden)
        a = torch.where(hidden > 0.5, one, hidden)
        a = torch.where(a < 0.5, zero, a)
        return a

    def predict_for_d(self, prev_list, bf, tau = 0.01):
        """
        :param prev_list:
        :param bf:
        :return:
        """
        input = bf
        one = self.bf_layer(input).unsqueeze(-1)
        two = self.bf_plus(one)
        two = F.gumbel_softmax(two, tau = tau)
        final = two[0,:,:,0].unsqueeze(0)
        return final

    def forward_bf(self, bf, tau = 0.8):
        """
        :param bf: [B, 340]
        :param tau: for gumbel softmax
        :return: prob of current act
        """
        """
        # half-half method, not so good.
        input = bf
        left_part = self.bf_layer(input).unsqueeze(-1)
        right_part = self.bf_layer_right(input).unsqueeze(-1)
        comb = torch.cat((left_part, right_part), -1)
        output = F.gumbel_softmax(comb, tau= tau)
        left_part_gumbel = output[0,:,:, 0].squeeze(-1).unsqueeze(0)
        print_test = left_part_gumbel[0].tolist()
        return left_part_gumbel
        """
        input = bf
        one = self.bf_layer(input).unsqueeze(-1)
        two = self.bf_plus(one)
        two = F.gumbel_softmax(two, tau = tau)
        final = two[0, :, :, 0].unsqueeze(0)
        return final

    def forward_embedding(self, prev_list, bf):
        """
        :param prev_list: list of tensors: [[ 1, 1, 549], [ 1, 2, 549]]
        :param hidden:
        :return: the probability of each slot.
        """
        input_embedding = tensor([])
        with torch.no_grad():
            for i, prev in enumerate(prev_list):
                # test = torch.ones(size=(1, 1, 549)).to(device).float()
                # emb_test = self.embeddings.compress(test)
                sum = torch.sum(prev)
                emb = self.embeddings.compress(prev).unsqueeze(0)
                input_embedding = torch.cat((input_embedding, emb), 1)
        # [ embedding: bf_embedding]
        # no bf embedding is better
        input = torch.cat((input_embedding, bf),2)
        hidden = self.emb_bf_layer(input)
        return hidden

    def sample(self, input_samples):
        """
        :param input_samples: list of tensors
        :param num_samples: number of samples
        :return: pos and neg, list of tensors
        """
        """
        1. get fake action: [1 , num, 549]
        2. concatenate to real and fake dialogue
        3. do embedding to every dialogue
        4. out put.
        """
        # set clip
        samples_pos = tensor([])
        samples_neg = tensor([])

        # make a sample work
        # with torch.no_grad():
        prev_list = []
        bf_temp = tensor([])
        target_temp = tensor([])
        for i, ele in enumerate(input_samples):
            prev, bf, target = input_samples[i]
            prev_list.append(prev)
            bf_temp = torch.cat((bf_temp, bf), 1)
            target_temp = torch.cat((target_temp, target), 1)
        # [1, num, 549]
        fake_action = self.predict_act(prev_list, bf_temp)
        """
        # for VAE embedding part
        for i, ele in enumerate(input_samples):
        prev, bf, target = ele
        # last [bf : action]
        last_real_cur = torch.cat((bf, target), 2)
        last_fake_cur = torch.cat((bf, fake_action[0][i].unsqueeze(0).unsqueeze(0)), 2)
        # whole dialogue [ 1, num, 1024]
        real_cur = torch.cat((prev.to("cuda"), last_real_cur.to("cuda")), 1)
        fake_cur = torch.cat((prev.to("cuda"), last_fake_cur.to("cuda")), 1)
        # do embedding
        emb = self.embeddings.compress(real_cur).unsqueeze(0).to("cuda")
        samples_pos = torch.cat((samples_pos, self.embeddings.compress(real_cur).unsqueeze(0).to("cuda")), 1)
        samples_neg = torch.cat((samples_neg, self.embeddings.compress(fake_cur).unsqueeze(0).to("cuda")), 1)
        """
        # for MLP part
        samples_pos = torch.cat((bf_temp, target_temp), 2)
        samples_neg = torch.cat((bf_temp, fake_action), 2)

        return samples_pos, samples_neg

    def batchNLLLoss(self, prev_list, bf, target, loss_func = "bce"):
        """
        :param input: bf [ , , ]
        :param target:   [ , , ]
        :return: loss
        """
        if loss_func == "bce":
            pos_weights = torch.full([209], 3, dtype=torch.float).to(DEVICE)
            # classification_loss = torch.nn.BCEWithLogitsLoss(reduction="sum", weight=pos_weights)
            classification_loss = torch.nn.BCELoss(reduction="sum")
            # regression_loss = torch.nn.MSELoss(reduction="sum")
            pred_action = self.forward_bf(bf)
            # diff = torch.sum(pred_action - target)
            loss = classification_loss(pred_action, target)
            pass
            # loss = classification_loss(self.forward_embedding(prev_list, bf), target)
        elif loss_func == "reg":
            # Todo： Write down the gumbel softmax function
            pass
        return loss

    def batchEVAL(self, prev_list, bf, target, forward_func = "bf"):
        """
        :param input: bf [ , , ]
        :param target:   [ , , ]
        :return: loss
        """
        action_prob = self.forward_bf(bf)
        # action_prob = self.forward_embedding(prev_list, bf.to("cuda"))
        # action = self.output_layer(action_prob)
        zero = torch.zeros_like(action_prob)
        one = torch.ones_like(action_prob)
        a = torch.where(action_prob > 0.5, one, action_prob)
        a = torch.where(a < 0.5, zero, a)
        test_loss_soft = torch.sum(torch.abs(a - target.to("cuda"))).to("cuda")
        a_1 = a.squeeze(0)
        t_1 = torch.transpose(target, 1, -1).squeeze(0)
        test_loss_hard = torch.sum(torch.matmul(a_1, t_1))
        return test_loss_soft, test_loss_hard

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

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
            if int(mask[i]) == 0:
                # for the last one, the reward should follow the system. 5, 40, -1, that's it.
                last_reward = r[i].item()
                reward_collc = self.get_score_g(s_train, a_train, tau=0.8)
                reward_collc[-1] += (last_reward+1)
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

    def get_score_g(self, input_bf, input_a, tau):
        """
        :param input: [ D, 340]
        :return:
        """
        # []
        assert input_bf.size(1)==input_a.size(1)
        prob_a = self.forward_bf(input_bf, tau=0.8)
        prob_a = prob_a.squeeze(0)
        real_a = input_a.clone().squeeze(0)
        reward = []
        for i in range(prob_a.size(0)):
            left = prob_a[i].unsqueeze(0)
            right = real_a[i].unsqueeze(-1)
            product = torch.matmul(left, right)
            reward.append(product.item() - 1)

        return reward
