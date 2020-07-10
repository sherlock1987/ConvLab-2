import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self, input_hidden_size = 1024):
        super(Discriminator, self).__init__()
        # self.d_layer = nn.Sequential(nn.Linear(input_hidden_size, np.ceil(input_hidden_size / 3)),
        #                        nn.ReLU(),
        #                        nn.Linear(np.ceil(input_hidden_size / 3), np.ceil(input_hidden_size / 6)),
        #                        nn.ReLU(),
        #                        nn.Linear(np.ceil(input_hidden_size / 6), np.ceil(input_hidden_size / 12)),
        #                        nn.ReLU(),
        #                        nn.Linear(np.ceil(input_hidden_size / 12), np.ceil(input_hidden_size / 1)))

        self.d_layer = nn.Sequential(nn.Linear(input_hidden_size, input_hidden_size//3),
                               nn.ReLU(),
                                     nn.Linear(input_hidden_size//3, input_hidden_size // 6),
                               nn.ReLU(),
                                     nn.Linear(input_hidden_size//6, input_hidden_size // 12),
                               nn.ReLU(),
                                     nn.Linear(input_hidden_size//12, 1),
                               nn.ReLU())
        # self.d_layer = nn.Sequential(nn.Linear(input_hidden_size,500),
        #                              nn.ReLU(),
        #                              nn.Linear(500,1))

        self.output_layer = nn.Sigmoid()

    def forward(self, input_hidden):
        prob = self.d_layer(input_hidden)
        output = self.output_layer(prob)
        return output


    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

