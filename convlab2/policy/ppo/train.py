# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import random

from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
# from convlab2.policy.ppo import PPO
from convlab2.policy.ppo.idea4.ppo import PPO
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()

def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()

def update(env, policy, batchsz, epoch, process_num):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    batchsz_real = s.size(0)
    policy.update(epoch, batchsz_real, s, a, r, mask)

# set seed, do not change the line stuff.
seed=2
# set seed, not over here.
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("current seed is {}".format(seed))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    # parser.add_argument("--save_path", type=str, default="test_1", help="path of model to save")
    # parser.add_argument("--save_st_path", type = int, default=0, help="sub path of model to save")
    parser.add_argument("--load_path_reward_d", default="", help="path of model to load from reward machine")
    parser.add_argument("--load_path_reward_g", default="", help="path of model to load from reward machine")
    parser.add_argument("--batchsz", type=int, default=40, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=40 , help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=1, help="number of processes of trajactory sampling")
    args = parser.parse_args()
    # sub_root = "convlab2/policy/ppo/idea4"
    # save_path = os.path.join(root_dir, sub_root, args.save_path, str(args.save_st_path))
    # print(save_path)
    # if not os.path.exists(save_path): os.makedirs(save_path)
    # simple rule DST
    dst_sys = RuleDST()

    policy_sys = PPO(True)
    # policy_sys.load('/home/raliegh/图片/ConvLab-2/convlab2/policy/mle/multiwoz/best_mle')
    #policy_sys.load_reward_model('/dockerdata/siyao/ft_local/ConvLab/convlab2/policy/mle/idea4/bin/idea4.pol.mdl')
    # policy_sys.load_reward_model_idea3(args.load_path_reward)
    # policy_sys.load(args.load_path)
    policy_sys.load_reward_model_idea4_d(args.load_path_reward_d)
    policy_sys.load_reward_model_idea4_g(args.load_path_reward_g)
    # should load three models.

    # not use dst
    dst_usr = None
    # rule policy
    policy_usr = RulePolicy(character='usr')
    # assemble
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    evaluator = MultiWozEvaluator()
    env = Environment(None, simulator, None, dst_sys)

    for i in range(args.epoch):
        update(env, policy_sys, args.batchsz, i, args.process_num)

"""
How to add idea_x?
args:
--load_path /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/multiwoz/best_mle --load_path_reward /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/multiwoz/save/idea1_model/idea_3_descriminator.mdl
idea4 model
--load_path /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/multiwoz/best_mle --load_path_reward_d /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/idea4/GAN1/Dis/pretrain_D.mdl --load_path_reward_g /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/idea4/GAN1/Gen/pretrain_G.mdl

then add init stuff in ppo.init
then add reward function in ppo, implement it in ppo.upgrade. # A_sa, v_target = self.est_adv(r, v, mask)
then add function of load model in ppo, implement it in ppo/train.py, modify args of load_reward_model
then run the code, remember you have to check if there is two models loaded, one is mle, one is reward model.

add reward model for idea4
1. Modify the args path
--load_path /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/multiwoz/best_mle --load_path_reward /home/raliegh/图片/ConvLab-2/convlab2/policy/mle/idea7/idea7_domain_tiny_data.pol.mdl
2. Modify ppo update func
3. Modify the import idea7
3. Untill you see that it could load two models
"""
