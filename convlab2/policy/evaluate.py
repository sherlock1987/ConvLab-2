# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.session import BiSession
from convlab2.dialog_agent.env import Environment
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import json
import matplotlib.pyplot as plt
import sys
import logging
import os
import datetime
import argparse
import os


def init_logging(log_dir_path, path_suffix=None):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if path_suffix:
        log_file_path = os.path.join(log_dir_path, f"{current_time}_{path_suffix}.log")
    else:
        log_file_path = os.path.join(log_dir_path, "{}.log".format(current_time))

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)
    format_str = "%(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
    logging.basicConfig(level=logging.DEBUG, handlers=[stderr_handler, file_handler], format=format_str)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def evaluate(dataset_name, model_name, load_path, calculate_reward=True):
    seed = 20190827
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dataset_name == 'MultiWOZ':
        dst_sys = RuleDST()

        if model_name == "PPO":
            from convlab2.policy.ppo import PPO
            if load_path:
                policy_sys = PPO(False)
                policy_sys.load(load_path)
            else:
                policy_sys = PPO.from_pretrained()
        elif model_name == "PG":
            from convlab2.policy.pg import PG
            if load_path:
                policy_sys = PG(False)
                policy_sys.load(load_path)
            else:
                policy_sys = PG.from_pretrained()
        elif model_name == "MLE":
            from convlab2.policy.mle.multiwoz import MLE
            if load_path:
                policy_sys = MLE()
                policy_sys.load(load_path)
            else:
                policy_sys = MLE.from_pretrained()
        elif model_name == "GDPL":
            from convlab2.policy.gdpl import GDPL
            if load_path:
                policy_sys = GDPL(False)
                policy_sys.load(load_path)
            else:
                policy_sys = GDPL.from_pretrained()

        dst_usr = None

        policy_usr = RulePolicy(character='usr')
        simulator = PipelineAgent(None, None, policy_usr, None, 'user')

        env = Environment(None, simulator, None, dst_sys)

        agent_sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')

        evaluator = MultiWozEvaluator()
        sess = BiSession(agent_sys, simulator, None, evaluator)

        task_success = {'All': []}
        for seed in range(100):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sess.init_session()
            sys_response = []
            logging.info('-' * 50)
            logging.info(f'seed {seed}')
            for i in range(40):
                sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
                if session_over is True:
                    task_succ = sess.evaluator.task_success()
                    # over here is our result, we need.
                    logging.info(f'task success: {task_succ}')
                    logging.info(f'book rate: {sess.evaluator.book_rate()}')
                    logging.info(f'inform precision/recall/f1: {sess.evaluator.inform_F1()}')
                    logging.info('-' * 50)
                    break
            else:
                task_succ = 0

            for key in sess.evaluator.goal:
                if key not in task_success:
                    task_success[key] = []
                else:
                    task_success[key].append(task_succ)
            task_success['All'].append(task_succ)

        for key in task_success:
            logging.info(
                f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}')
        if calculate_reward:
            reward_tot = []
            for seed in range(100):
                s = env.reset()
                reward = []
                value = []
                mask = []
                for t in range(40):
                    s_vec = torch.Tensor(policy_sys.vector.state_vectorize(s))
                    a = policy_sys.predict(s)

                    # interact with env
                    next_s, r, done = env.step(a)
                    logging.info(r)
                    reward.append(r)
                    if done:  # one due to counting from 0, the one for the last turn
                        break
                logging.info(f'{seed} reward: {np.mean(reward)}')
                reward_tot.append(np.mean(reward))
            logging.info(f'total avg reward: {np.mean(reward_tot)}')
    else:
        raise Exception("currently supported dataset: MultiWOZ")


# this is my code of changing
def evaluate_test(dataset_name, model_name, load_path, calculate_reward=True):
    """
    :param dataset_name:
    :param model_name:
    :param load_path:
    :param calculate_reward:
    :return: dict: task_success
    """
    seed = 20190827
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dataset_name == 'MultiWOZ':
        dst_sys = RuleDST()

        if model_name == "PPO":
            from convlab2.policy.ppo import PPO
            if load_path:
                policy_sys = PPO(False)
                policy_sys.load(load_path)
            else:
                policy_sys = PPO.from_pretrained()
        elif model_name == "PG":
            from convlab2.policy.pg import PG
            if load_path:
                policy_sys = PG(False)
                policy_sys.load(load_path)
            else:
                policy_sys = PG.from_pretrained()
        elif model_name == "MLE":
            from convlab2.policy.mle.multiwoz import MLE
            if load_path:
                policy_sys = MLE()
                policy_sys.load(load_path)
            else:
                policy_sys = MLE.from_pretrained()
        elif model_name == "GDPL":
            from convlab2.policy.gdpl import GDPL
            if load_path:
                policy_sys = GDPL(False)
                policy_sys.load(load_path)
            else:
                policy_sys = GDPL.from_pretrained()

        dst_usr = None

        policy_usr = RulePolicy(character='usr')
        simulator = PipelineAgent(None, None, policy_usr, None, 'user')

        env = Environment(None, simulator, None, dst_sys)

        agent_sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')

        evaluator = MultiWozEvaluator()
        sess = BiSession(agent_sys, simulator, None, evaluator)

        task_success = {'All': []}
        for seed in range(100):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sess.init_session()
            sys_response = []
            logging.info('-' * 50)
            logging.info(f'seed {seed}')
            for i in range(40):
                sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
                if session_over is True:
                    task_succ = sess.evaluator.task_success()
                    # over here is our result, we need.
                    logging.info(f'task success: {task_succ}')
                    logging.info(f'book rate: {sess.evaluator.book_rate()}')
                    logging.info(f'inform precision/recall/f1: {sess.evaluator.inform_F1()}')
                    logging.info('-' * 50)
                    break
            else:
                task_succ = 0

            for key in sess.evaluator.goal:
                if key not in task_success:
                    task_success[key] = []
                else:
                    task_success[key].append(task_succ)
            task_success['All'].append(task_succ)

        for key in task_success:
            logging.info(
                f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}')
        if calculate_reward:
            reward_tot = []
            for seed in range(100):
                s = env.reset()
                reward = []
                value = []
                mask = []
                for t in range(40):
                    s_vec = torch.Tensor(policy_sys.vector.state_vectorize(s))
                    a = policy_sys.predict(s)

                    # interact with env
                    next_s, r, done = env.step(a)
                    logging.info(r)
                    reward.append(r)
                    if done:  # one due to counting from 0, the one for the last turn
                        break
                logging.info(f'{seed} reward: {np.mean(reward)}')
                reward_tot.append(np.mean(reward))
            logging.info(f'total avg reward: {np.mean(reward_tot)}')
        return task_success
    else:
        raise Exception("currently supported dataset: MultiWOZ")


def create_file_list(dir):
    path = dir
    files = os.listdir(path)
    name = []
    num = []
    # detect the last one
    files.sort()

    for file in files:
        if os.path.isfile(path+"/"+file):
            if file[:3].isnumeric():
                num.append(int(file[:3]))
            elif file[:2].isnumeric():
                num.append(int(file[:2]))
            else:
                num.append(int(file[0]))

    num = list(set(num))
    num.sort()
    print(num)
    if args.model_name == "PPO":
        for nums in num:
            name.append(dir + "/" + str(nums) + "_ppo")
    elif args.model_name == "GDPL":
        for nums in num:
            name.append(dir + "/" + str(nums) + "_gdpl")
    else:
        pass
    return name


sample_dict = {
    'All': [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    'train': [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
              1, 0, 0],
    'restaurant': [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1,
                   0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    'attraction': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'hotel': [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'taxi': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    'hospital': [1]}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="MultiWOZ", help="name of dataset")
    parser.add_argument("--model_name", type=str, default="PPO", help="name of model")
    parser.add_argument("--load_path", type=str, default='', help="path of model")
    parser.add_argument("--log_path_suffix", type=str, default="", help="suffix of path of log file")
    parser.add_argument("--log_dir_path", type=str, default="log", help="path of log directory")
    parser.add_argument("--evluate_in_dir", type=bool, default=False,
                        help="whether to evluate the model from one directory")
    parser.add_argument("--model_path_root", type=str, default="", help="root path which contains many models")
    args = parser.parse_args()

    init_logging(log_dir_path=args.log_dir_path, path_suffix=args.log_path_suffix)
    success_rate_record = []
    if not args.evluate_in_dir:
        evaluate(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            load_path=args.load_path,
            calculate_reward=True
        )
    else:
        name_list = create_file_list(args.model_path_root)
        print(name_list)
        for model_dir in name_list:
            print("Evaluating the model from", model_dir)
            result = evaluate_test(
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                load_path=model_dir,
                calculate_reward=False
            )
            success_rate = np.mean(result["All"])
            success_rate_record.append(success_rate)
        #  remember to print otherwise you will lose time.
        print(success_rate_record)
        axis = [i for i in range(len(success_rate_record))]
        plt.plot(axis, success_rate_record)
        plt.xlabel('Number of Epoch')
        plt.ylabel('Success rate')
        plt.show()
# args for the evluator
# --load_path
# /home/raliegh/视频/ConvLab-2/convlab2/policy/pg/save/0_pg_plus_reward.pol.mdl
# --model_name
# GDPL
# --model_path_root
# /home/raliegh/视频/ConvLab-2/convlab2/policy/gdpl/save/original
# --evluate_in_dir
# True

#  seems like training on the best one is not a good idea at all.
#  reward + best one
# [0.32, 0.22, 0.2, 0.2, 0.2]
# Only best one
# [0.29, 0.33, 0.07, 0.0] 训练出现错误了.
#  reward + MLE
# [0.44, 0.3, 0.38, 0.36, 0.36]
# MLE
# [0.23, 0.0]
# Nothing
# [0.22, 0.23, 0.26, 0.28, 0.26]
# reaward + no best one or MLE
# [0.33, 0.37, 0.3, 0.29, 0.36]

# PPO
# MLE
# [0.56, 0.59, 0.57, 0.6, 0.63, 0.62, 0.65, 0.68, 0.67, 0.7, 0.71, 0.72, 0.73, 0.71, 0.72, 0.72, 0.73, 0.76, 0.7, 0.74, 0.82, 0.77, 0.77, 0.76, 0.77, 0.73, 0.77, 0.71, 0.71, 0.7]
# MLE + idea5(local_real)
# [0.59, 0.56, 0.62, 0.65, 0.66, 0.67, 0.7, 0.68, 0.7, 0.73, 0.74, 0.69, 0.74, 0.77, 0.73, 0.75, 0.71, 0.74, 0.74, 0.76, 0.77, 0.75, 0.73, 0.73, 0.69, 0.72, 0.7, 0.72, 0.74, 0.75]
# MLE + idea5(global_mask)
# [0.57, 0.6, 0.65, 0.64, 0.67, 0.7, 0.73, 0.77, 0.73, 0.72, 0.73, 0.73, 0.72, 0.7, 0.72, 0.72, 0.71, 0.73, 0.7, 0.71, 0.72, 0.74, 0.73, 0.74, 0.73, 0.75, 0.76, 0.76, 0.74, 0.77]
# MLE + idea5(local + global(cos))
# [0.57, 0.63, 0.67, 0.67, 0.67, 0.66, 0.65, 0.63, 0.69, 0.72, 0.69, 0.69, 0.7, 0.71, 0.72, 0.73, 0.61, 0.64, 0.72, 0.74, 0.69, 0.73, 0.69, 0.72, 0.73, 0.73, 0.72, 0.67, 0.67, 0.71]
# MLE + idea5(local + global(mask))
# [0.57, 0.63, 0.65, 0.66, 0.7, 0.71, 0.69, 0.72, 0.74, 0.73, 0.71, 0.73, 0.74, 0.75, 0.75, 0.74, 0.73, 0.68, 0.74, 0.7, 0.73, 0.72, 0.72, 0.73, 0.74, 0.73, 0.73, 0.76, 0.75, 0.74]
# MLE + idea6(local domain classify)
# [0.57, 0.58, 0.61, 0.69, 0.67, 0.67, 0.71, 0.68, 0.68, 0.7, 0.74, 0.68, 0.73, 0.68, 0.68, 0.71, 0.71, 0.68, 0.76, 0.71, 0.7, 0.7, 0.69, 0.67, 0.69, 0.72, 0.68, 0.73, 0.69, 0.68]
# MLE + idea6(local domain + global(mask)) better than only local.
# [0.59, 0.67, 0.66, 0.69, 0.72, 0.73, 0.71, 0.71, 0.73, 0.75, 0.72, 0.74, 0.7, 0.75, 0.72, 0.72, 0.7, 0.73, 0.73, 0.73, 0.73, 0.73, 0.72, 0.73, 0.72, 0.73, 0.74, 0.74, 0.75, 0.75]
# MLE + idea6(local domain + global(mask))+updating
# [0.57, 0.63, 0.67, 0.72, 0.75, 0.74, 0.74, 0.76, 0.77, 0.78, 0.79, 0.77, 0.78, 0.78, 0.79, 0.76, 0.74, 0.74, 0.74, 0.76, 0.76, 0.79, 0.78, 0.73, 0.75, 0.78, 0.75, 0.7, 0.73, 0.73]
# MLE + reward drop significantly 预测的不对,基本上很难有所帮助的
# [0.52, 0.53, 0.54, 0.49, 0.44, 0.49, 0.46, 0.47, 0.44, 0.43, 0.42, 0.44, 0.47, 0.48, 0.49, 0.46, 0.45, 0.46, 0.45, 0.48, 0.46, 0.48, 0.49, 0.49, 0.49, 0.48, 0.45, 0.47, 0.43, 0.43, 0.43, 0.42, 0.42, 0.41, 0.42, 0.43, 0.44, 0.47, 0.45, 0.43]
# best model
# [0.69, 0.73, 0.76, 0.75, 0.71, 0.67, 0.68, 0.68, 0.68, 0.71, 0.71, 0.68, 0.7, 0.67, 0.7, 0.7, 0.64, 0.66, 0.65, 0.63]
# MLE + reward_3 40
# extremelly quick when using this one, much more stable than the previous one, and also the rewards2 have some bugs, remember to modify it.
# [0.52, 0.57, 0.55, 0.58, 0.57, 0.55, 0.54, 0.58, 0.55, 0.53, 0.55, 0.53, 0.54, 0.55, 0.57, 0.53, 0.55, 0.54, 0.56, 0.56, 0.55, 0.54, 0.57, 0.56, 0.58, 0.6, 0.6, 0.55, 0.57, 0.57, 0.58, 0.58, 0.59, 0.57, 0.57, 0.57, 0.57, 0.57, 0.56, 0.58]
""""
Description for how to use this file
args:
1. Evluate all of models in one root
PPO
--model_name PPO --evluate_in_dir True --model_path_root /home/raliegh/图片/ConvLab-2/convlab2/policy/ppo/idea5/save
GDPL
--model_name GDPL --evluate_in_dir True --model_path_root /home/raliegh/图片/ConvLab-2/convlab2/policy/gdpl/save

2. Evluate only one model
--model_name PPO --load_path /home/raliegh/视频/ConvLab-2/convlab2/policy/pg/save/0_pg_plus_reward.pol.mdl
Then run, remember to see model load successfully in logs.
"""""
# PPO 1024 MLE + domain
# [0.62, 0.62, 0.68, 0.69, 0.72, 0.73, 0.7, 0.74, 0.74, 0.78, 0.77, 0.71, 0.72, 0.77, 0.77, 0.77, 0.8, 0.75, 0.77, 0.75, 0.75, 0.75, 0.74, 0.78, 0.77, 0.75, 0.78, 0.75, 0.75, 0.78]
# PPO 1024 MLE
# [0.58, 0.64, 0.65, 0.66, 0.67, 0.66, 0.64, 0.68, 0.68, 0.69, 0.7, 0.69, 0.69, 0.69, 0.69, 0.67, 0.69, 0.68, 0.69, 0.69, 0.69, 0.68, 0.68, 0.68, 0.63, 0.66, 0.61, 0.63, 0.63, 0.66]