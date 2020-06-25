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
from convlab2.policy.mle.Fake_data_generator import PG_generator
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.tensor as tensor
# result for the descriminator
res = []
print(len(res))
record = []
temp = 0
for i in range(len(res)):
    if (i%100) == 0:
        record.append(temp/100)
        temp = 0
    if res[i]==1:
        temp+=1
print(record)
axis = [i for i in range(len(record))]
plt.plot(axis, record)
plt.xlabel('Number of dialogues')
plt.ylabel('Embedding Loss')
plt.show()
