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

# -*- coding: utf-8 -*-
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

class PG(Policy):
    def __init__(self, is_train=False, dataset='Multiwoz'):
        with open("/home/raliegh/图片/ConvLab-2/convlab2/policy/pg/config.json", 'r') as f:
            cfg = json.load(f)
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        self.save_per_epoch = cfg['save_per_epoch']
        self.update_round = cfg['update_round']
        self.optim_batchsz = cfg['batchsz']
        self.gamma = cfg['gamma']
        self.is_train = is_train
        if is_train:
            init_logging_handler(cfg['log_dir'])

        if dataset == 'Multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file)
            self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)

        # self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)
        if is_train:
            self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=cfg['lr'])

        # define the predictor params

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (tensor): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        with torch.no_grad():
            a = self.policy.select_action(state.to(device=DEVICE), self.is_train).cpu()
            action = self.vector.action_devectorize(a.detach().numpy())
            state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass

    def load(self, filename):
        policy_mdl_candidates = [
            filename,
            filename + '.pol.mdl',
            filename + '_pg.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_pg.pol.mdl')
        ]
        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                break

# If need to generate the fake data, please write the code
fake_policy = PG()
state =tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.])
action = fake_policy.predict(state)
# print(action)