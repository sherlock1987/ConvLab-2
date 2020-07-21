import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict


def expierment_name(args, ts):

    exp_name = str()
    exp_name += "TS=%s"%ts
    return exp_name
