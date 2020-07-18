#!/usr/bin/env Python
# coding=utf-8
import torch
from torch.autograd import Variable
from math import ceil
from random import randint, sample


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target

def filter(pos_samples, neg_samples):
    """
    :param pos_samples:
    :param neg_samples:
    :return: fiter the same [bf: act] pair
    """
    # first size is same.
    num_samples = pos_samples.size(1)
    pos_samples = pos_samples.squeeze(0)
    neg_samples = neg_samples.squeeze(0)
    list_stay = []
    for i in range(num_samples):
        pos_tiny = pos_samples[i]
        neg_tiny = neg_samples[i]
        if torch.equal(pos_tiny, neg_tiny):
        #     print("find same, extinguish in neg")
            pass
        else:
            list_stay.append(i)
    neg_samples = neg_samples[list_stay].unsqueeze(0)
    pos_samples = pos_samples.unsqueeze(0)
    return pos_samples, neg_samples

def prepare_discriminator_data(pos_samples, neg_samples, ratio_pos, ratio_neg, gpu=False):
    """
    :param pos_samples: [1, num_samples, 549]
    :param neg_samples:[1, num_samples, 549]

    :param gpu: True
    :return: random samples and target(1, 0)
    """
    # get number for each case.
    pos_samples, neg_samples = filter(pos_samples, neg_samples)
    # stay same number
    pos_samples = pos_samples[0][:neg_samples.size(1)].unsqueeze(0)
    if ratio_pos > ratio_neg:
        # 1 : 0.9
        pos_num = pos_samples.size(1)
        neg_num = int(neg_samples.size(1)* (ratio_neg/ratio_pos))
    elif ratio_pos == ratio_neg:
        # 1 : 1
        pos_num = pos_samples.size(1)
        neg_num = neg_samples.size(1)
    else:
        # 1 : 10
        pos_num = int(pos_samples.size(1) * (ratio_pos/ratio_neg))
        neg_num = neg_samples.size(1)

    pos_samples_clone = pos_samples.clone()
    pos_samples_clone = pos_samples_clone[0][:pos_num].unsqueeze(0)
    neg_samples_clone = neg_samples.clone()
    neg_samples_clone = neg_samples_clone[0][:neg_num].unsqueeze(0)
    inp = torch.cat((pos_samples_clone, neg_samples_clone), 1).type(torch.LongTensor)
    target = torch.ones(pos_samples_clone.size(1) + neg_samples_clone.size(1))
    # [real : fake] = [1 : 0]
    target[pos_samples_clone.size(1):] = 0.
    # shuffle
    perm = torch.randperm(target.size(0))
    target = target[perm]
    inp = inp[0][perm].unsqueeze(0)
    # set variable to unchanged, and have grad
    inp = Variable(inp)
    target = Variable(target)
    if gpu:
        inp = inp.cuda()
        target = target.cuda()
    # Todo: what about extra dataset.
    return inp, target


def batchwise_sample(gen, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)
def prepare_data(data , gpu):
    """
    :param input: dict: test: list of tensors
    :return:
    """
    for key, value in data.items:

        print()
def oracle_sample(input_data, num):
    """
    :param input_data: list of tensors
    :return: list of tensors
    """
    return sample(input_data, num)

