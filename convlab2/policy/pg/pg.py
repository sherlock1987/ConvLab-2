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

class Reward_predict(nn.Module):

    def __init__(self,input_size, hidden_size):
        super(Reward_predict,self).__init__()
        self.cnn = nn.Linear(input_size, hidden_size, bias=True)
        self.encoder = nn.LSTM(hidden_size,hidden_size,batch_first=True,bidirectional=False)
        self.cnn2 = nn.Linear(hidden_size,hidden_size,bias=True)
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, input,target):
        # to construct the batch first, then we could compute the loss function for this stuff, simple and easy.
        feature_input = self.cnn(input)
        _, (predictor_vec, last_cell) = self.encoder(feature_input)
        loss = self.loss(predictor_vec,target,target=torch.tensor(1))
        return loss

    def target_extract(self,target):
        with torch.no_grad():
            feature_target = self.cnn(target)
            _, (encoded_target, last_cell) = self.encoder(feature_target)
            return encoded_target

class PG(Policy):
    def __init__(self, is_train=False, dataset='Multiwoz'):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
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
        self.predictor_reward = Reward_predict(549,457)
        self.optim_reward = optim.Adam(self.predictor_reward.parameters(), lr=1e-1 )
        self.loss_record = []

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.policy.select_action(s_vec.to(device=DEVICE), self.is_train).cpu()
        action = self.vector.action_devectorize(a.numpy())
        state['system_action'] = action

        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass

    def est_return(self, r, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: V-target(s), Tensor
        """
        batchsz = r.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)

        prev_v_target = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # update previous
            prev_v_target = v_target[t]

        return v_target

    def update(self, epoch, batchsz, s, a, r, mask):
        # generate a sequence of this one, it is fine for me
        v_target = self.est_return(r, mask)

        # from mask to do the divide and get the index
        batch_index = []
        temp = []
        for index, ele in enumerate(mask.numpy()):
            if ele == 0:
                temp.append(index)
                batch_index.append(temp)
                temp = []
            else:
                temp.append(index)

        for i in range(self.update_round):
            for iteration in batch_index:
                loss = torch.tensor([0]).float()
                # starting looping and make the sum of loss.
                print("-" * 30,len(iteration))
                for index, _ in enumerate(iteration):
                    temp = iteration[:index + 1]
                    if len(temp) >= 4:
                        # start training
                        s_b = s[temp].float()
                        a_b = a[temp].float()
                        input = torch.cat((s_b[:-1], a_b[:-1]), -1).unsqueeze(0)
                        target = torch.cat((s_b[-1], a_b[-1]), -1).unsqueeze(0).unsqueeze(0)
                        target = self.predictor_reward.target_extract(target)

                        micro_loss = self.predictor_reward(input, target)
                        # print(micro_loss)
                        loss += micro_loss
                # backwarding...
                if loss != torch.tensor([0]).float():
                    print(loss,loss.shape)
                    loss.backward()
                    for name,param in self.predictor_reward.named_parameters():
                        if "cnn" not in name:
                            # print(name)
                            # print(param.grad)
                            pass
                    self.optim_reward.step()
                    self.predictor_reward.zero_grad()
                    self.loss_record.append(loss.item())
            # # 1. shuffle current batch
            # perm = torch.randperm(batchsz)
            # # shuffle the variable for mutliple optimize
            # # v_target_shuf, s_shuf, a_shuf = v_target[perm], s[perm], a[perm]
            # v_target_shuf, s_shuf, a_shuf = v_target, s, a
            # # 2. get mini-batch for optimizing
            # optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            # # chunk the optim_batch for total batch
            # v_target_shuf, s_shuf, a_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
            #                                 torch.chunk(s_shuf, optim_chunk_num), \
            #                                 torch.chunk(a_shuf, optim_chunk_num)
            #
            # # 3. iterate all mini-batch to optimize
            # policy_loss = 0.
            # for v_target_b, s_b, a_b in zip(v_target_shuf, s_shuf, a_shuf):
            #     # print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
            #
            #     # update policy network by clipping
            #     self.policy_optim.zero_grad()
            #     # [b, 1]
            #     log_pi_sa = self.policy.get_log_prob(s_b, a_b)
            #     # ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
            #     # we use log_pi for stability of numerical operation
            #     # [b, 1] => [b]
            #     # this is element-wise comparing.
            #     # we add negative symbol to convert gradient ascent to gradient descent
            #     surrogate = - (log_pi_sa * v_target_b).mean()
            #     policy_loss += surrogate.item()
            #
            #     # backprop
            #     surrogate.backward()
            #     # gradient clipping, for stability
            #     torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
            #     # self.lock.acquire() # retain lock to update weights
            #     self.policy_optim.step()
            #     # self.lock.release() # release lock
            #
            # policy_loss /= optim_chunk_num
            # logging.debug('<<dialog policy pg>> epoch {}, iteration {}, policy, loss {}'.format(epoch, i, policy_loss))

        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_pg.pol.mdl')
        torch.save(self.predictor_reward.state_dict(), directory + '/' + str(epoch) + '_pg.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))
        axis = [i for i in range(len(self.loss_record))]
        plt.plot(axis,self.loss_record)
        plt.xlabel('Number of dialogue turns')
        plt.ylabel('Embedding Loss')
        plt.show()

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

    def load_from_pretrained(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for PG Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_pg.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_pg.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))

    @classmethod
    def from_pretrained(cls,
                        archive_file="",
                        model_file="https://convlab.blob.core.windows.net/convlab-2/pg_policy_multiwoz.zip"):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        model = cls()
        model.load_from_pretrained(archive_file, model_file, cfg['load'])
        return model