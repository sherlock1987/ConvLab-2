import os
import tqdm as tqdm
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.tensor as tensor
import torch

import copy
import numpy as np
import pandas as pd
from arff2pandas import a2p
import os
import torch
from convlab2.policy.mle.multiwoz.loader import ActMLEPolicyDataLoaderMultiWoz
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.mle.fake_data_generator import PG_generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "/home/raliegh/图片/ConvLab-2/convlab2/policy/mle/processed_data/"


def get_data(part_which):
    """
    This is a function used for create dataset.
    :param part_which:train val test
    :return:
    """
    terminate = {}
    state_whole = {}
    input = {}
    # load data of terminate
    for part in ['train', 'val', 'test']:
        with open(os.path.join("//home//raliegh//图片//ConvLab-2//convlab2//policy//mle//multiwoz//processed_data",
                               '{}_terminate.pkl'.format(part)), 'rb') as f:
            terminate[part] = pickle.load(f)
    # load data of dict
    for part in ['train', 'val', 'test']:
        with open(os.path.join("//home//raliegh//图片//ConvLab-2//convlab2//policy//mle//multiwoz//processed_data",
                               '{}_state_whole.pkl'.format(part)), 'rb') as f:
            state_whole[part] = pickle.load(f)

    manager = ActMLEPolicyDataLoaderMultiWoz()


    data_whole= manager.create_dataset(part_which, 1)
    s_temp = torch.tensor([])
    a_temp = torch.tensor([])

    # make some batch
    # [[ 1, 5, 549],]
    data_list = []
    assert (len(data_whole) == len(terminate[part_which]))
    print("this data contains about {} turns in toral".format(len(data_whole)))
    for i, data in enumerate(data_whole):
        s, a = to_device(data)
        try:
            s_temp = torch.cat((s_temp, s), 0)
            a_temp = torch.cat((a_temp, a), 0)
        except Exception as e:
            s_temp = s
            a_temp = a
        # [ , , ]
        s_train = s_temp.unsqueeze(0)
        a_train = a_temp.unsqueeze(0)
        if len(s_train[0]) >= 2:
            input_real = torch.cat((s_train, a_train), 2)
            flag = terminate[part_which][i]

            if flag == False:
                pass
            else:
                data_list.append(input_real)
                s_temp = torch.tensor([])
                a_temp = torch.tensor([])
                input_real = torch.tensor([])
    input[part_which] = data_list
    pickle.dump(input, open(os.path.join(save_path, 'sa_{}.pkl'.format(part_which)), 'wb'))

def get_elementwise_data(part_which):
    """
    This is a function used for create dataset.
    :param part_which:train val test
    :return:
    """
    terminate = {}
    state_whole = {}
    input = {}
    # load data of terminate
    with open(os.path.join(save_path,
                           '{}_terminate.pkl'.format(part_which)), 'rb') as f:
        terminate[part_which] = pickle.load(f)
    # load data of dict
    with open(os.path.join(save_path,
                           '{}_state_whole.pkl'.format(part_which)), 'rb') as f:
        state_whole[part_which] = pickle.load(f)
    # /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/processed_data/train_terminate.pkl
    manager = ActMLEPolicyDataLoaderMultiWoz()


    data_whole= manager.create_dataset(part_which, 1)
    s_temp = torch.tensor([])
    a_temp = torch.tensor([])

    # make some batch
    # [[ 1, 5, 549],]
    data_list = []
    assert (len(data_whole) == len(terminate[part_which]))
    print("this data contains about {} dialog in total".format(len(data_whole)))
    for i, data in enumerate(data_whole):
        s, a = to_device(data)
        try:
            s_temp = torch.cat((s_temp, s), 0)
            a_temp = torch.cat((a_temp, a), 0)
        except Exception as e:
            s_temp = s
            a_temp = a
        # [ , , ]
        s_train = s_temp.unsqueeze(0)
        a_train = a_temp.unsqueeze(0)

        input_real = torch.cat((s_train, a_train), 2)
        flag = terminate[part_which][i]

        if flag == False:
            data_list.append(input_real)
        else:
            data_list.append(input_real)
            s_temp = torch.tensor([])
            a_temp = torch.tensor([])
            input_real = torch.tensor([])

    input[part_which] = data_list
    pickle.dump(input, open(os.path.join(save_path, 'sa_element_{}_real.pkl'.format(part_which)), 'wb'))

def get_elementwise_fake(part_which):
    """
    This is a function used for create dataset.
    :param part_which:train val test
    :return:
    """
    terminate = {}
    state_whole = {}
    input = {}
    # load data of terminate
    with open(os.path.join(save_path,
                           '{}_terminate.pkl'.format(part_which)), 'rb') as f:
        terminate[part_which] = pickle.load(f)
    # load data of dict
    with open(os.path.join(save_path,
                           '{}_state_whole.pkl'.format(part_which)), 'rb') as f:
        state_whole[part_which] = pickle.load(f)
    # /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/processed_data/train_terminate.pkl
    manager = ActMLEPolicyDataLoaderMultiWoz()


    data_whole= manager.create_dataset(part_which, 1)
    s_temp = torch.tensor([])
    a_temp = torch.tensor([])

    # [[ 1, 5, 549],]
    data_list = []
    assert (len(data_whole) == len(terminate[part_which]))
    assert (len(data_whole) == len(state_whole[part_which]))

    fake_generater = PG_generator()
    print("this data contains about {} dialog in total".format(len(data_whole)))
    for i, data in enumerate(data_whole):
        s, a = to_device(data)
        state = state_whole[part_which][i]
        fake_data = fake_generater.predict(state)
        try:
            s_temp = torch.cat((s_temp, s), 0)
            a_temp = torch.cat((a_temp, a), 0)
        except Exception as e:
            s_temp = s
            a_temp = a
        # [ , , ]
        s_train = s_temp.unsqueeze(0)
        a_train = a_temp.unsqueeze(0)

        input_real = torch.cat((s_train, a_train), 2)
        flag = terminate[part_which][i]

        if flag == False:
            # change this to fake action
            input_fake = input_real.clone()

            input_fake[0][-1][340:549] = fake_data
            # test if this will work.
            # a_test = input_real[0][-1][340:549]
            # b_test = input_fake[0][-1][340:549]
            data_list.append(input_fake)
        else:
            # change this to fake action
            input_fake = input_real.clone()
            input_fake[0][-1][340:549] = fake_data
            data_list.append(input_fake)
            s_temp = torch.tensor([])
            a_temp = torch.tensor([])
            input_real = torch.tensor([])

    input[part_which] = data_list
    pickle.dump(input, open(os.path.join(save_path, 'sa_element_{}_fake.pkl'.format(part_which)), 'wb'))

def load_data(part_which):
    """
    This is a function used for load dataset.
    :param part_which: which part should we consider
    :return: data_list, terminate, dict at each state.
    """
    state_whole = {}
    input = {}
    terminate = {}
    with open(os.path.join(save_path, 'sa_{}.pkl'.format(part_which)), 'rb') as f:
        input["data"] = pickle.load(f)
    data_list = input["data"]

    # load dict
    with open(os.path.join("//home//raliegh//图片//ConvLab-2//convlab2//policy//mle//multiwoz//processed_data",
                           '{}_state_whole.pkl'.format(part_which)), 'rb') as f:
        state_whole[part_which] = pickle.load(f)
    # load terminate
    with open(os.path.join("//home//raliegh//图片//ConvLab-2//convlab2//policy//mle//multiwoz//processed_data",
                           '{}_terminate.pkl'.format(part_which)), 'rb') as f:
        terminate[part_which] = pickle.load(f)

    print("finish load {} data".format(part_which))
    return data_list[part_which], state_whole[part_which], terminate[part_which]

# how to use?
# get_data(part_which="test")
# data_list, state_whole, terminate = load_data(part_which="train")
get_elementwise_fake("train")
get_elementwise_fake("val")
get_elementwise_fake("test")
