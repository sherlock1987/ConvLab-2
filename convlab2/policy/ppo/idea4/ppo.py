# -*- coding: utf-8 -*-
from importlib import reload
import torch
from torch import optim
from torch import tensor
import numpy as np
import logging
import os
import json
from convlab2.policy.policy import Policy
from convlab2.policy.rlmodule import MultiDiscretePolicy, Value
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
import zipfile
import sys
from convlab2.policy.mle.idea4.GAN1.discriminator import Discriminator
from convlab2.policy.mle.idea4.GAN1.generator import Generator


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

class PPO(Policy):
    def __init__(self, is_train=False, dataset='Multiwoz'):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        print("---")
        print(self.save_dir)
        print("---")
        self.save_per_epoch = cfg['save_per_epoch']
        self.update_round = cfg['update_round']
        self.optim_batchsz = cfg['batchsz']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.tau = cfg['tau']
        self.is_train = is_train
        if is_train:
            init_logging_handler(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['log_dir']))

        # construct policy and value network
        if dataset == 'Multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file)
            self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)

        self.value = Value(self.vector.state_dim, cfg['hv_dim']).to(device=DEVICE)
        if is_train:
            self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=cfg['policy_lr'])
            self.value_optim = optim.Adam(self.value.parameters(), lr=cfg['value_lr'])

        self.reward_predictor_idea4_d = Discriminator()
        self.reward_predictor_idea4_g = Generator()

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.policy.select_action(s_vec.to(device=DEVICE), False).cpu()
        action = self.vector.action_devectorize(a.numpy())
        state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass

    def est_adv(self, r, v, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)
        delta = torch.Tensor(batchsz).to(device=DEVICE)
        A_sa = torch.Tensor(batchsz).to(device=DEVICE)

        prev_v_target = 0
        prev_v = 0
        prev_A_sa = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_A_sa = A_sa[t]
        # normalize A_sa
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()

        return A_sa, v_target

    def reward_estimate(self, r, s, a, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :param s: state, Tensor, [b,340]
        :param a: action, Tensor, [b,209]
        """
        reward_predict = []
        batchsz = r.shape[0]
        s_temp = torch.tensor([])
        a_temp = torch.tensor([])
        for i in range(batchsz):
            # current　states and actions
            s_1 = s[i].unsqueeze(0)
            a_1 = a[i].unsqueeze(0)
            try:
                s_temp = torch.cat((s_temp, s_1), 0)
                a_temp = torch.cat((a_temp, a_1), 0)
            except Exception:
                s_temp = s_1
                a_temp = a_1

            s_train = s_temp.unsqueeze(0).float()
            a_train = a_temp.unsqueeze(0).float()
            input = torch.cat((s_train, a_train), 2)
            # input_bf = s_train[-1].unsqueeze(0)
            # target = a_train[-1].unsqueeze(0)
            # 长度大于2 的话呢.
            if len(input[0]) >= 2:
                input_pre = input.squeeze(0)[:-1].unsqueeze(0)
                input_bf = s_train.squeeze(0)[-1].unsqueeze(0).unsqueeze(0)
                target = a_train.squeeze(0)[-1].unsqueeze(0).unsqueeze(0)
                # print(input_pre.shape,input_bf.shape,target.shape)
                if int(mask[i]) == 0:
                    # for the last one, the reward should follow the system.
                    cur_reward = r[i]
                    s_temp = torch.tensor([])
                    a_temp = torch.tensor([])
                #   compute the last one, terminate clear the button, that is okay for us.
                else:
                    with torch.no_grad():
                        cur_reward = self.reward_predictor.compute_reward(input_pre, input_bf, target)
                reward_predict.append(cur_reward.item())
            else:
                # when the lengh is 1, the start reward is 1
                reward_predict.append(1)

        reward_predict = tensor(reward_predict)
        return reward_predict

    def reward_estimate_idea5(self, r, s, a, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :param s: state, Tensor, [b,340]
        :param a: action, Tensor, [b,209]
        """
        reward_predict = self.reward_predictor_idea5.get_reward(r, s, a, mask)
        return reward_predict

    def reward_estimate_idea4(self, r, s, a, mask):
        """
        get reward from G and D.
        :param r: reward, Tensor, [b]
        :param s: state, Tensor, [b,340]
        :param a: action, Tensor, [b,209]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        """
        reward_predict_d = self.reward_predictor_idea4_d.get_reward(r, s, a, mask)
        # reward_predict_g = self.reward_predictor_idea4_g.get_reward(r, s, a, mask)
        # reward_predict = (reward_predict_d+reward_predict_g)/2
        return reward_predict_d

    def update(self, epoch, batchsz, s, a, r, mask):
        # get estimated V(s) and PI_old(s, a)
        # actually, PI_old(s, a) can be saved when interacting with env, so as to save the time of one forward elapsed
        # v: [b, 1] => [b]
        v = self.value(s).squeeze(-1).detach()
        log_pi_old_sa = self.policy.get_log_prob(s, a).detach()

        # estimate advantage and v_target according to GAE and Bellman Equation
        # leave the V alone, just forget about it.
        # r = self.reward_estimate(r, s, a, mask)
        # r = self.reward_estimate_idea4(r, s, a, mask)
        A_sa, v_target = self.est_adv(r, v, mask)

        for i in range(self.update_round):

            # 1. shuffle current batch
            perm = torch.randperm(batchsz)
            # shuffle the variable for mutliple optimize
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
                                                                           log_pi_old_sa[perm]

            # 2. get mini-batch for optimizing, this size comes from config.
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            # chunk the optim_batch for total batch
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
                                                                           torch.chunk(A_sa_shuf, optim_chunk_num), \
                                                                           torch.chunk(s_shuf, optim_chunk_num), \
                                                                           torch.chunk(a_shuf, optim_chunk_num), \
                                                                           torch.chunk(log_pi_old_sa_shuf,
                                                                                       optim_chunk_num)
            # 3. iterate all mini-batch to optimize
            policy_loss, value_loss = 0., 0.
            for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
                                                                     log_pi_old_sa_shuf):
                # print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
                # 1. update value network
                self.value_optim.zero_grad()
                v_b = self.value(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()

                # backprop
                loss.backward()
                # nn.utils.clip_grad_norm(self.value.parameters(), 4)
                self.value_optim.step()

                # 2. update policy network by clipping
                self.policy_optim.zero_grad()
                # [b, 1]
                log_pi_sa = self.policy.get_log_prob(s_b, a_b)
                # ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
                # we use log_pi for stability of numerical operation
                # [b, 1] => [b]
                ratio = (log_pi_sa - log_pi_old_sa_b).exp().squeeze(-1)
                # because the joint action prob is the multiplication of the prob of each da
                # it may become extremely small
                # and the ratio may be inf in this case, which causes the gradient to be nan
                # clamp in case of the inf ratio, which causes the gradient to be nan
                ratio = torch.clamp(ratio, 0, 10)
                surrogate1 = ratio * A_sa_b
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
                # this is element-wise comparing.
                # we add negative symbol to convert gradient ascent to gradient descent
                surrogate = - torch.min(surrogate1, surrogate2).mean()
                policy_loss += surrogate.item()

                # backprop
                surrogate.backward()
                # although the ratio is clamped, the grad may still contain nan due to 0 * inf
                # set the inf in the gradient to 0
                for p in self.policy.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                # gradient clipping, for stability
                torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
                # self.lock.acquire() # retain lock to update weights
                self.policy_optim.step()
                # self.lock.release() # release lock

            value_loss /= optim_chunk_num
            policy_loss /= optim_chunk_num
            logging.debug('<<dialog policy ppo>> epoch {}, iteration {}, value, loss {}'.format(epoch, i, value_loss))
            logging.debug('<<dialog policy ppo>> epoch {}, iteration {}, policy, loss {}'.format(epoch, i, policy_loss))

        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)

    def save(self, directory, epoch):
        # print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.value.state_dict(), directory + '/' + str(epoch) + '_ppo.val.mdl')
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_ppo.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename):
        flag = 1
        value_mdl_candidates = [
            filename + '.val.mdl',
            filename + '_ppo.val.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.val.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_ppo.val.mdl')
        ]
        for value_mdl in value_mdl_candidates:
            if os.path.exists(value_mdl):
                self.value.load_state_dict(torch.load(value_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))
                flag = 1
                break

        policy_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_ppo.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_ppo.pol.mdl')
        ]
        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                flag = 1
                break
        assert flag==1

    def load_reward_model(self, filename):
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
                self.reward_predictor.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded reward model checkpoint from file: {}'.format(policy_mdl))
                break

    def load_reward_model_idea4_d(self, filename):
        flag = 0
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
                self.reward_predictor_idea4_d.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                self.reward_predictor_idea4_d = self.reward_predictor_idea4_d.to(DEVICE)
                logging.info('<<dialog policy>> loaded reward_idea4 D model checkpoint from file: {}'.format(policy_mdl))
                flag = 1
                break
        assert flag==1

    def load_reward_model_idea4_g(self, filename):
        flag = 0
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
                self.reward_predictor_idea4_g.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                self.reward_predictor_idea4_g = self.reward_predictor_idea4_g.to(DEVICE)
                logging.info('<<dialog policy>> loaded reward_idea4 G model checkpoint from file: {}'.format(policy_mdl))
                flag = 1
                break
        assert flag==1

    def load_from_pretrained(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for PPO Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_ppo.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_ppo.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))

        value_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_ppo.val.mdl')
        if os.path.exists(value_mdl):
            self.value.load_state_dict(torch.load(value_mdl, map_location=DEVICE))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))

    @classmethod
    def from_pretrained(cls,
                        archive_file="",
                        model_file="https://convlab.blob.core.windows.net/convlab-2/ppo_policy_multiwoz.zip",
                        is_train=False,
                        dataset='Multiwoz'):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        model = cls(is_train=is_train, dataset=dataset)
        model.load_from_pretrained(archive_file, model_file, cfg['load'])
        return model