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
from tqdm import tqdm

# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
from torch import optim
from torch import tensor
import numpy as np
import logging
import os
import json
from convlab2.policy.policy import Policy
from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
import zipfile
import sys
import matplotlib.pyplot  as plt
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim=64):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.cnn = nn.Linear(in_features = input_size, out_features = input_size)
        # self.rnn1 = nn.LSTM(input_size = embedding_dim, hidden_size = self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, seq_len, x):
        self.seq_len = seq_len
        # x = x.reshape((1, self.seq_len, self.input_size))
        x = self.cnn(x)
        x, (hidden, _) =   self.rnn1(x)
        # double layer lstm x: [ , , ]
        x, (hidden_n, _) = self.rnn2(x)
        # hidden_n: [1, 1, 209]
        # x: [1, 5, 209]
        # [1, 1, 418]
        return hidden_n

class Decoder(nn.Module):
    def __init__(self, input_dim=64, input_size=1):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.input_size = input_size
        self.rnn1 = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        # self.rnn1 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, input_size)
        self.rescaler = nn.Softmax(dim=1)
        # self.rescaler = nn.Sigmoid()


    def forward(self, seq_len, x):
        self.seq_len = seq_len
        # make broadcast then put into LSTM
        x = x.repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        # x, (hidden_n, cell_n) = self.rnn1(x)
        # [1, 1, 418]
        output = self.output_layer(x)
        return self.rescaler(output[0])

class RecurrentAutoencoder(nn.Module):
    def __init__(self, input_size, embedding_dim= 1098):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(input_size, embedding_dim).to(DEVICE)
        self.decoder = Decoder(embedding_dim, input_size).to(DEVICE)

    def forward(self, x):
        seq_len = len(x[0])
        x = self.encoder(seq_len, x)
        x = self.decoder(seq_len, x)
        return x

    def compress(self,input):
        with torch.no_grad():
            seq_len = len(input[0])
            hidden_states = self.encoder(seq_len, input)
            # [209]
            return hidden_states.squeeze(0).squeeze(0)

def auto_encoder(data_train):
    model = RecurrentAutoencoder(549,1098)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)
    # criterion_cosine = nn.CosineEmbeddingLoss(reduction="sum")
    model = model.train()
    epoch = 3
    train_loss_whole = []
    for j in range(epoch):
        train_losses = []
        # for i in tqdm(range(len(data_train))):
        for i in (range(10)):
            seq_true = data_train[i]
            optimizer.zero_grad()
            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)
            # seq_true = seq_true + 1e-4
            # seq_pred = seq_pred + 1e-4
            # print(torch.sum(seq_pred, dim = 1))

            loss = criterion(seq_pred, seq_true.squeeze(0))
            # loss = criterion(seq_pred, seq_true, target = tensor(1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item()/len(seq_true[0]))
            train_loss_whole.append(loss.item()/len(seq_true[0]))
            # for name, param in model.named_parameters():
            #     if "output_layer" not in name:
            #         print(name)
            #         print(param.grad)
            #         pass
        train_loss = np.mean(train_losses)
        # my_y_ticks = np.arange(0, 2, 0.05)
        print("current epoch is :"+str(j)," loss is: ",train_loss)
    # save current model.
    torch.save(model.state_dict(),"/home/raliegh/图片/ConvLab-2/convlab2/policy/mle/" + "auto_encoder.pol.mdl")

    my_y_ticks = np.arange(0, 500, 50)
    axis = [i for i in range(len(train_loss_whole))]
    plt.plot(axis, train_loss_whole)
    plt.yticks(my_y_ticks)
    plt.show()