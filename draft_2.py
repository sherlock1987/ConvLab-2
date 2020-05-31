import torch
import torch.nn as nn
from torch import optim
from torch import tensor
import numpy as np
import logging
import os
import json
import zipfile
import sys

class Reward_predict(nn.Module):

    def __init__(self,input_size, hidden_size):
        super(Reward_predict,self).__init__()
        self.cnn = nn.Linear(input_size, hidden_size, bias=True)
        self.encoder = nn.LSTM(hidden_size,hidden_size,batch_first=True,bidirectional=False)
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, input,target):
        # to construct the batch first, then we could compute the loss function for this stuff, simple and easy.
        feature_input = self.cnn(input)
        _, (predictor_vec, last_cell) = self.encoder(feature_input)
        # feature_extract = self.cnn_2(predictor_vec)
        # loss = self.loss_2(feature_extract[0][0],tensor([20]).float())
        loss = self.loss(predictor_vec,target,target=tensor(1))
        return loss

    def target_extract(self,target):
        with torch.no_grad():
            feature_target = self.cnn(target)
            _, (encoded_target, last_cell) = self.encoder(feature_target)
            return encoded_target

mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
        1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.,
        1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
        1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 0.])
batch = []
temp = []
for index,ele in enumerate(mask.numpy()):
    if ele == 0:
        temp.append(index)
        batch.append(temp)
        temp = []
    else:
        temp.append(index)

s = torch.rand((582,340))
a = torch.rand((582,209))
r = torch.rand((582))
# 直接生成一个这么个东西.
input_size =  int(s.shape[1]+a.shape[1])
hidden_size = int(input_size // 1.2)

model = Reward_predict(input_size,hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.00001)


for iteration in batch:
    # start training
    print("-"*30)
    for i,index in enumerate(iteration):
        temp = iteration[:i+1]
        if len(temp) != 1:
            # start training
            # optimizer.zero_grad()
            s_b = s[temp]
            a_b = a[temp]
            input = torch.cat((s_b[:-1],a_b[:-1]),-1).unsqueeze(0)
            # target = torch.rand((hidden_size)).unsqueeze(0).unsqueeze(0)
            target = torch.rand((input_size)).unsqueeze(0).unsqueeze(0)

            target = model.target_extract(target)
            try:
                loss += model(input,target)
            except Exception as e:
                loss = model(input,target)
    loss.backward()
    # for name,param in model.named_parameters():
    #     pass
    #     print(name)
    #     print(param.grad)
    optimizer.step()
    model.zero_grad()
    loss = torch.tensor([0]).float()
