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
        self.bf_bf = nn.Linear(340, 340)
        self.output_layer = nn.Sigmoid()
    def load_VAE(self, path):
        self.embeddings.load_state_dict(torch.load(path, map_location=DEVICE))

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
    3.1 First I will try predict 300 actions.
    3.2 Next I will try add embedding
    3.3 Next I will try 
    """
    def predict_act(self, prev_list, bf):
        hidden = self.forward_embedding(prev_list, bf)
        hidden_prob = self.output_layer(hidden).to("cuda")
        # make action looks like real
        zero = torch.zeros_like(hidden_prob)
        one = torch.ones_like(hidden_prob)
        a = torch.where(hidden_prob > 0.5, one, hidden_prob)
        a = torch.where(a < 0.5, zero, a)

        return a

    def forward_bf(self, bf):
        """
        :param prev_list: list of tensors: [[ 1, 1, 549], [ 1, 2, 549]]
        :param hidden:
        :return: the probability of each slot.
        """
        input = bf
        hidden = self.bf_layer(input)
        return hidden

    def forward_embedding(self, prev_list, bf):
        """
        :param prev_list: list of tensors: [[ 1, 1, 549], [ 1, 2, 549]]
        :param hidden:
        :return: the probability of each slot.
        """
        input_embedding = tensor([])
        with torch.no_grad():
            for i, prev in enumerate(prev_list):
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
        test = torch.ones(size = (1,1,549)).to(device).float()
        emb_test = self.embeddings.compress(test)
        test_list = [test]
        self.forward_embedding(test_list, torch.zeros(size=(1,1,340)).to(device))
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

        return samples_pos, samples_neg

    def batchNLLLoss(self, prev_list, bf, target, loss_func = "bce"):
        """
        :param input: bf [ , , ]
        :param target:   [ , , ]
        :return: loss
        """
        if loss_func == "bce":
            pos_weights = torch.full([209], 3, dtype=torch.float).to(DEVICE)
            classification_loss = torch.nn.BCEWithLogitsLoss(reduction="sum", weight=pos_weights)
            # loss = classification_loss(self.forward_embedding(prev_list, bf), target)
            loss = classification_loss(self.forward_embedding(prev_list, bf), target)
        elif loss_func == "gumbel_softmax":
            # Todo： Write down the gumbel softmax function
            pass
        return loss

    def batchEVAL(self, prev_list, bf, target, forward_func = "bf"):
        """
        :param input: bf [ , , ]
        :param target:   [ , , ]
        :return: loss
        """
        # action_prob = self.forward_bf(bf)
        action_prob = self.forward_embedding(prev_list, bf.to("cuda"))
        action = self.output_layer(action_prob)
        test_loss = torch.sum(torch.abs((action > 0.5).type(tensor) - target.to("cuda"))).to("cuda")
        return test_loss

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

