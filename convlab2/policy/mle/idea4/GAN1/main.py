#!/usr/bin/env Python
# coding=utf-8
"""
Potential Bugs:
    1. check if embedding is all zero.
    2. Check the predict action.
"""
from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
from convlab2.policy.mle.idea4.GAN1.utils import expierment_name
import torch
import torch.optim as optim
import torch.nn as nn
import os
from convlab2.policy.mle.idea4.GAN1.generator import Generator
from convlab2.policy.mle.idea4.GAN1.discriminator import Discriminator
import convlab2.policy.mle.idea4.GAN1.helpers
from convlab2.policy.mle.idea4.GAN1.helpers import prepare_data, prepare_discriminator_data, oracle_sample, batchwise_sample, batchwise_oracle_nll
import pickle
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device != "cuda"
import time
import collections
import argparse
from tensorboardX import SummaryWriter

# set  seed
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Global Variable
GAN_step = 0
D_train_step = 0
G_train_step = 0
tracker = collections.defaultdict(list)
# path stuff
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_path = os.path.join(root_dir, "mle/processed_data")
# raise NotImplementedError
# convlab2/policy/mle/idea4/model/VAE_20_123.pol.mdl
# wrong VAE model, GRU not work at all
# load_VAE_path = os.path.join(root_dir, "mle/idea4/model/VAE_20_123.pol.mdl")
load_VAE_path = os.path.join(root_dir, "mle/idea4/model/VAE_39_complete.pol.mdl")

# path stuff
pretrained_gen_path = os.path.join(root_dir, "mle/idea4/GAN1/Gen")
pretrained_dis_path = os.path.join(root_dir, "mle/idea4/GAN1/Dis")
if not os.path.exists(pretrained_gen_path): os.makedirs(pretrained_gen_path)
if not os.path.exists(pretrained_dis_path): os.makedirs(pretrained_dis_path)

tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def train_generator_MLE(gen, gen_opt, real_data_samples, epochs = 10):
    """
    :param gen:
    :param gen_opt:
    :param real_data_samples:
    :param epochs:
    :return:
    """
    """
    1. --Train, val at each epoch.
    2. Test at last epoch.
    3. Change two stuff for quicker watching, 1. seq_train, 2. dataset
    4. No sample, since this is MLE training process.
    5. 30 epoch seems not enough. the loss is still going downer
    """
    seq_train = ["train", "val", "test"]
    # seq_train = ["val", "train", "test"]
    for epoch in range(epochs):
        gen.train()
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        prev_list = []
        bf_temp = torch.tensor([]).to(DEVICE)
        target_temp = torch.tensor([]).to(DEVICE)
        for iteration, ele in enumerate(real_data_samples[seq_train[0]]):
            # from here to get input, and target.
            prev, bf, target = real_data_samples[seq_train[0]][iteration]
            prev_list.append(prev.to(DEVICE))
            bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
            target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

            if len(prev_list) == args.BATCH_SIZE_G:
                gen_opt.zero_grad()
                loss = gen.batchNLLLoss(prev_list, bf_temp, target_temp)
                loss.backward()
                # for name, param in gen.named_parameters():
                #     print(name)
                #     print(param.grad)
                gen_opt.step()
                total_loss += loss.data.item()

                # empty the clip
                prev_list = []
                bf_temp = tensor([])
                target_temp = tensor([])

            if ceil((iteration / args.BATCH_SIZE_G) % 100) == 0:
                print('.', end='')
                sys.stdout.flush()
        total_loss = total_loss / float(iteration)
        writer.add_scalar("G_MLE/train_loss", total_loss, epoch)
        print('average_train_Loss = %.4f' % (total_loss), end="   ")
        """
        Validation part:
        Evaluation: minus, product, average action.
        val_eval- = 2.4540, val_eval* = 761.4751 (2.56 act)
        
        """
        # do validation loss over here.
        gen.eval()
        sys.stdout.flush()
        prev_list = []
        bf_temp = tensor([])
        target_temp = tensor([])
        for iteration, ele in enumerate(real_data_samples["val"]):
            # from here to get input, and target.
            prev, bf, target = real_data_samples["val"][iteration]
            prev_list.append(prev.to(DEVICE))
            bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
            target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

        target_score = torch.sum(target_temp)
        val_eval_1, val_eval_2 = gen.batchEVAL(prev_list, bf_temp, target_temp)
        # empty the clip
        prev_list = []
        bf_temp = tensor([])
        target_temp = tensor([])

        val_eval_1 = val_eval_1.item() / float(iteration)
        val_eval_2 = val_eval_2.item() / float(iteration)
        target_score = target_score.item() / float(iteration)
        print('val_eval- = %.4f, val_eval* = %.4f (%.2f act)' % (val_eval_1, val_eval_2, target_score))
        writer.add_scalar("G_MLE/val_eval-", val_eval_1, epoch)
        writer.add_scalar("G_MLE/val_eval*", val_eval_2, epoch)

    # do test loss over here.
    gen.eval()
    sys.stdout.flush()
    prev_list = []
    bf_temp = tensor([])
    target_temp = tensor([])
    for iteration, ele in enumerate(real_data_samples["test"]):
        # from here to get input, and target.
        prev, bf, target = real_data_samples["test"][iteration]
        prev_list.append(prev.to(DEVICE))
        bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
        target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

    target_score = torch.sum(target_temp)
    val_eval_1, val_eval_2 = gen.batchEVAL(prev_list, bf_temp, target_temp)
    # empty the clip
    prev_list = []
    bf_temp = tensor([])
    target_temp = tensor([])

    val_eval_1 = val_eval_1.item() / float(iteration)
    val_eval_2 = val_eval_2.item() / float(iteration)
    target_score = target_score.item() / float(iteration)
    print('test_eval- = %.4f, test_eval* = %.4f (%.2f act)' % (val_eval_1, val_eval_2, target_score))
    writer.add_scalar("G/test_eval-", val_eval_1, 0)
    writer.add_scalar("G/test_eval*", val_eval_2, 0)

def train_generator_PG(gen, dis, real_data_samples, G_D_lr = 1e-4, num_samples = 5000, num_val_sample = 2000, epochs=10):
    """
    :param gen:
    :param gen_opt:
    :param oracle:
    :param dis:
    :param epochs:
    :return:
    """
    """
    1. Do validation to see if generator went wrong.
    2.     Train and Validation.
    """
    # do validation in the first place.
    gen.eval()
    prev_list = []
    bf_temp = torch.tensor([], requires_grad=False).to(DEVICE)
    target_temp = torch.tensor([]).to(DEVICE)
    global GAN_step, G_train_step
    val_oracle = oracle_sample(real_data_samples["val"], num_val_sample)

    for iteration, ele in enumerate(val_oracle):
        # from here to get input, and target.
        prev, bf, target = val_oracle[iteration]
        prev_list.append(prev.to(DEVICE))
        bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
        target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

    pred_action = gen.predict_for_d(prev_list, bf_temp, tau=0.001)
    one_a = pred_action[0][-1]
    dis_input = torch.cat((bf_temp, pred_action), 2)
    # max reward = min -reward (0 ~ 1)
    reward = dis(dis_input)
    reward = torch.sum(reward)
    # should be no gradient
    # for name, param in gen.named_parameters():
    #     print(name)
    #     print(param.grad)
    val_loss = reward.data.item()
    val_loss /= float(iteration)
    print('Val_Reward = %.4f' % (val_loss))
    writer.add_scalar("G/val_reward_1", val_loss, GAN_step)
    GAN_step += 1
    # empty the clip
    prev_list = []
    bf_temp = tensor([])
    target_temp = tensor([])

    # sample some stuff.
    data_cur = real_data_samples["train"]
    # data_cur = real_data_samples["val"]
    sample_oracle_data = oracle_sample(data_cur, num_samples)
    # set dis no gradient
    dis.eval()
    gen_optimizer = optim.Adam(gen.parameters(), lr=G_D_lr)
    for epoch in range(epochs):
        gen.train()
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        prev_list = []
        bf_temp = torch.tensor([], requires_grad=False).to(DEVICE)
        target_temp = torch.tensor([]).to(DEVICE)
        for iteration, ele in enumerate(sample_oracle_data):
            # from here to get input, and target.
            prev, bf, target = sample_oracle_data[iteration]
            prev_list.append(prev.to(DEVICE))
            bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
            target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

            if len(prev_list) == args.BATCH_SIZE_G:
                gen_optimizer.zero_grad()
                pred_action = gen.predict_for_d(prev_list, bf_temp, tau = 0.01)
                one_a = pred_action[0][-1]
                dis_input = torch.cat((bf_temp, pred_action), 2)
                # max reward = min -reward (0 ~ 1)
                reward = - dis(dis_input)
                reward = torch.sum(reward)
                reward.backward()
                # for name, param in gen.named_parameters():
                #     print(name)
                #     print(param.grad)
                gen_optimizer.step()

                total_loss += reward.data.item()
                # empty the clip
                prev_list = []
                bf_temp = tensor([])
                target_temp = tensor([])

            if ceil((iteration / args.BATCH_SIZE_G) % 100) == 0:
                print('.', end='')
                sys.stdout.flush()
        total_loss = - total_loss / float(iteration)
        # print('average_train_reward = %.4f' % (total_loss), end="   ")
        print('train_reward = %.4f' % (total_loss), end = "    ")

        # Evaluation
        gen.eval()
        prev_list = []
        bf_temp = torch.tensor([], requires_grad=False).to(DEVICE)
        target_temp = torch.tensor([]).to(DEVICE)
        val_oracle = oracle_sample(real_data_samples["val"], num_val_sample)

        for iteration, ele in enumerate(val_oracle):
            # from here to get input, and target.
            prev, bf, target = val_oracle[iteration]
            prev_list.append(prev.to(DEVICE))
            bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
            target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

        pred_action = gen.predict_for_d(prev_list, bf_temp, tau = 0.001)
        one_a = pred_action[0][-1]
        dis_input = torch.cat((bf_temp, pred_action), 2)
        # max reward = min -reward (0 ~ 1)
        reward = dis(dis_input)
        reward = torch.sum(reward)
        # should be no gradient
        # for name, param in gen.named_parameters():
        #     print(name)
        #     print(param.grad)
        val_loss = reward.data.item()
        val_loss /= float(iteration)
        print('val_reward = %.4f' % (val_loss))
        writer.add_scalar("G/val_reward", val_loss, G_train_step)
        G_train_step += 1
        # empty the clip
        prev_list = []
        bf_temp = tensor([])
        target_temp = tensor([])

def train_discriminator(discriminator, dis_opt, real_data_samples, generator, sample_num, ratio_pos = 1.0 ,ratio_neg = 1.0, d_steps = 1, epochs = 30):
    """
    :param discriminator:
    :param dis_opt:
    :param real_data_samples:
    :param generator:
    :param sample_num:
    :param ratio_pos: [pos_num: neg_num] = [ratio_pos: ratio_neg]
    :param ratio_neg: [pos_num: neg_num] = [ratio_pos: ratio_neg]
    :param d_steps:
    :param epochs:
    :return:
    """
    """
    1. Shuffle data first
    2. Take the positive to generate negative samples. [pos_num: neg_num] = [ratio_pos: ratio_neg]
    3. Using neg, pos to update D, the thing is, we only have one neg at one time if generator is not update.
    4. If D performs pretty good, we will update G at more time. I think so!!
    """
    # random dataset
    data_cur = real_data_samples["train"]
    # data_cur = real_data_samples["val"]
    random.shuffle(data_cur)
    # repeat this for d_steps
    global D_train_step
    generator.eval()
    for d_step in range(d_steps):
        # do sample first
        sample_oracle_data = oracle_sample(data_cur, sample_num)
        pos_train, neg_trian = generator.sample(sample_oracle_data)
        # [real: fake] = [1: 0] sigmoid--1
        train_inp, train_target = prepare_discriminator_data(pos_train, neg_trian, ratio_pos, ratio_neg, gpu=args.cuda)
        loss_fn = nn.MSELoss(reduction = "sum")
        for epoch in range(epochs):
            discriminator.train()
            print('Sample %d data, d-step %d epoch %d : ' % (train_inp.size(1), d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            # empty clip
            total_loss = 0
            total_acc = 0
            total_eval = 0
            total_pos_pred = 0
            total_neg_pred = 0

            # go to the batch
            batch_num = np.ceil((train_inp.size(1)) / args.BATCH_SIZE_D)
            for i in range(int(batch_num)):
                # prepare data
                pointer = i * args.BATCH_SIZE_D
                if i == int(batch_num - 1):
                    # for　the　last batch
                    inp, target = train_inp[0][pointer: -1].unsqueeze(0), train_target[pointer: -1].unsqueeze(0).unsqueeze(0)
                    target = target.view(1, -1, 1)
                else:
                    inp, target = train_inp[0][pointer: pointer + args.BATCH_SIZE_D].unsqueeze(0), train_target[pointer: pointer + args.BATCH_SIZE_D].unsqueeze(0).unsqueeze(0)
                    target = target.view(1, -1, 1)

                # step
                dis_opt.zero_grad()
                inp = inp.float()
                out = discriminator(inp)
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                # print model collapse
                # for name,param in discriminator.named_parameters():
                #     if torch.sum(param.grad).item() == 0.:
                #         print("name: {} is zero grad at epoch {} and batch {}".format(epoch, i, name))

                # push grad moving
                # for p in discriminator.parameters():
                #     p.grad[p.grad != p.grad] = 0.0

                # print the last one.
                if (epoch == epochs-1 and i == int(batch_num - 1)) or (epoch == 0 and i == int(0)) :
                    # print(out)
                    for name, param in discriminator.named_parameters():
                        pass
                        # print(name)
                        # print(param.grad)

                total_loss += loss.data.item()

                # check for + and -
                for i in range(target.size(1)):
                    out_1 = out[0][i].item()
                    tar_1 = target[0][i].item()
                    if tar_1 >0.5 and out_1>0.5:
                        total_pos_pred+=1
                    elif tar_1<0.5 and out_1<0.5:
                        total_neg_pred+=1
                    else:
                        pass

                total_acc += torch.sum((out>0.5) == (target>0.5)).data.item()
                # do evaluation over here.
                # . . .
                total_eval += torch.abs(torch.sum(out - target)).item()
                if (i / args.BATCH_SIZE_D ) % 10 == 0:
                    print('.', end='')
                    sys.stdout.flush()
            total_loss/=  (train_inp.size(1))

            if ratio_pos > ratio_neg:
                total_acc /=  (sample_num + sample_num*(ratio_neg /ratio_pos))
                total_pos_pred /= sample_num
                total_neg_pred /= (sample_num*(ratio_neg / ratio_pos))
                total_eval /= (sample_num + sample_num*(ratio_neg /ratio_pos))/10
            else:
                # more neg
                total_acc /=  (sample_num + sample_num*(ratio_pos /ratio_neg))
                total_pos_pred /= (sample_num*(ratio_pos /ratio_neg))
                total_neg_pred /= sample_num
                total_eval /= (sample_num + sample_num*(ratio_pos /ratio_neg))/10

            # do validation
            # data is from val
            val_pos_pred = 0
            val_neg_pred = 0
            discriminator.eval()
            num_val_sample = 1000
            sample_oracle_data = oracle_sample(real_data_samples["val"], num_val_sample)

            pos_val, neg_val = generator.sample(sample_oracle_data)
            val_inp, val_target = prepare_discriminator_data(pos_val, neg_val, ratio_pos, ratio_neg, gpu=args.cuda)
            # do loss again
            val_target = val_target.unsqueeze(0).unsqueeze(-1).view(1, -1, 1)
            val_pred = discriminator(val_inp.float())
            loss_val = loss_fn(val_pred, val_target)
            loss_val = loss_val.item()
            loss_val /= val_inp.size(1)

            val_eval = torch.abs(torch.sum(val_pred - val_target)).item()

            # check for + and -
            for i in range(val_target.size(1)):
                out_1 = val_pred[0][i].item()
                tar_1 = val_target[0][i].item()
                if tar_1 > 0.5 and out_1 > 0.5:
                    val_pos_pred += 1
                elif tar_1 < 0.5 and out_1 < 0.5:
                    val_neg_pred += 1
                else:
                    pass

            if ratio_pos > ratio_neg:
                total_acc /=  (num_val_sample + num_val_sample*(ratio_neg /ratio_pos))
                val_pos_pred /= num_val_sample
                val_neg_pred /= (num_val_sample*(ratio_neg / ratio_pos))
                val_eval /= (num_val_sample + num_val_sample*(ratio_neg /ratio_pos))/10

            else:
                # more neg
                total_acc /=  (num_val_sample + num_val_sample*(ratio_pos /ratio_neg))
                val_pos_pred /= (num_val_sample*(ratio_pos /ratio_neg))
                val_neg_pred /= num_val_sample
                val_eval /= (num_val_sample + num_val_sample*(ratio_pos /ratio_neg))/10

            print('Train: loss = %.4f, eval = %.4f, pos:neg=[%.2f : %.2f]    Val: loss = %.4f, eval = %.4f, pos:neg=[%.2f : %.2f]' % (total_loss, total_eval, total_pos_pred, total_neg_pred, loss_val, val_eval, val_pos_pred, val_neg_pred))
            writer.add_scalar("D/val_loss", total_loss, D_train_step)
            writer.add_scalar("D/val_eval", total_loss, D_train_step)
            writer.add_scalar("D/val_pos", total_loss, D_train_step)
            writer.add_scalar("D/val_neg", total_loss, D_train_step)
            D_train_step += 1

def test_g_d(gen, dis, real_data_samples):
    """
    :param gen: current G
    :param dis: current D
    :param real_data_samples: test data, all of them.
    :return: score of G(batchEVAL) and D(score)
    """
    global GAN_step
    GAN_step += 1
    gen.eval()
    sys.stdout.flush()
    prev_list = []
    bf_temp = tensor([])
    target_temp = tensor([])
    for iteration, ele in enumerate(real_data_samples["test"]):
        # from here to get input, and target.
        prev, bf, target = real_data_samples["test"][iteration]
        prev_list.append(prev.to(DEVICE))
        bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
        target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

    target_score = torch.sum(target_temp)
    val_eval_1, val_eval_2 = gen.batchEVAL(prev_list, bf_temp, target_temp)
    # empty the clip
    prev_list = []
    bf_temp = tensor([])
    target_temp = tensor([])

    val_eval_1 = val_eval_1.item() / float(iteration)
    val_eval_2 = val_eval_2.item() / float(iteration)
    target_score = target_score.item() / float(iteration)
    tracker["test_eval-"].append(val_eval_1)
    tracker["test_eval*"].append(val_eval_2)
    tracker["average_act"].append(target_score)
    print('TEST: G: test_eval- = %.4f, test_eval* = %.4f (%.2f act)' % (val_eval_1, val_eval_2, target_score), end = "  ")
    writer.add_scalar("G/test_eval-", val_eval_1, GAN_step)
    writer.add_scalar("G/test_eval*", val_eval_2, GAN_step)

    # testing D
    test_pos_pred = 0
    test_neg_pred = 0
    ratio_pos = 1.
    ratio_neg = 1.
    loss_fn = nn.MSELoss(reduction = "sum")
    dis.eval()
    sample_oracle_data = oracle_sample(real_data_samples["test"], len(real_data_samples["test"]))
    pos_val, neg_val = gen.sample(sample_oracle_data)
    test_inp, test_target = prepare_discriminator_data(pos_val, neg_val, ratio_pos, ratio_neg, gpu=args.cuda)
    # do loss again
    test_target = test_target.unsqueeze(0).unsqueeze(-1).view(1, -1, 1)
    test_pred = dis(test_inp.float())
    loss_test = loss_fn(test_pred, test_target).item()
    loss_test /= test_inp.size(1)

    test_eval = torch.abs(torch.sum(test_pred - test_target)).item()
    test_eval /= test_inp.size(1)/10
    # check for + and -
    for i in range(test_target.size(1)):
        out_1 = test_pred[0][i].item()
        tar_1 = test_target[0][i].item()
        if tar_1 > 0.5 and out_1 > 0.5:
            test_pos_pred += 1
        elif tar_1 < 0.5 and out_1 < 0.5:
            test_neg_pred += 1
        else:
            pass

    test_pos_pred /= (test_inp.size(1)/2)
    test_neg_pred /= (test_inp.size(1)/2)

    tracker["d_loss"].append(loss_test)
    tracker["d_eval"].append(test_eval)
    tracker["d_pos"].append(test_pos_pred)
    tracker["d_neg"].append(test_neg_pred)

    print('D: sample %d data, loss = %.4f, eval = %.4f, pos:neg=[%.2f : %.2f]' % (test_inp.size(1), loss_test, test_eval, test_pos_pred, test_neg_pred))
    writer.add_scalar("D/test_loss", loss_test, GAN_step)
    writer.add_scalar("D/test_eval", test_eval, GAN_step)
    writer.add_scalar("D/test_pos", test_pos_pred, GAN_step)
    writer.add_scalar("D/test_neg", test_neg_pred, GAN_step)

# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--BATCH_SIZE_G', type=int, default=32)
    parser.add_argument('--BATCH_SIZE_D', type=int, default=32)
    parser.add_argument('--logdir', type=str, default="logs")

    # epochs
    parser.add_argument('--MLE_TRAIN_EPOCHS', type=int, default=60)
    parser.add_argument('--ADV_TRAIN_EPOCHS', type=int, default=2)
    parser.add_argument('--save_epoch', type=int, default=5)

    # lr
    parser.add_argument('--gen_lr', type=float, default=1e-3)
    parser.add_argument('--gen_lr_ad', type=float, default=1e-3)
    parser.add_argument('--dis_lr', type=float, default=1e-3)
    args = parser.parse_args()

    ts = time.strftime('%Y-%b-%d-%H:%M:%S')
    gen = Generator(gpu = args)
    dis = Discriminator()
    if args.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
    # gen.load_VAE(load_VAE_path)
    print("load VAE model successfully")

    # set up writer
    writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
    writer.add_text("model_G", str(gen))
    writer.add_text("model_D", str(dis))
    writer.add_text("args", str(args))
    writer.add_text("ts", ts)

    # DATA READING
    print("Starting Loading Data")
    splits = ['train', 'val']
    # splits = ['val']
    if args.test: splits.append("test")
    datasets = {}
    for split in splits:
        with open(os.path.join(data_path, 'sa_prev_bf_a_{}.pkl'.format(split)), 'rb') as f:
            datasets[split] = pickle.load(f)[split]
    print("Data Loading Successful")
    print("Starting Processing Data into [prev: bf: a]")

    """
    GENERATOR MLE TRAINING
    """
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=args.gen_lr)
    # train_generator_MLE(gen, gen_optimizer, datasets, args.MLE_TRAIN_EPOCHS)
    # load again, clear optimizer
    # torch.save(gen.state_dict(), os.path.join(pretrained_gen_path, "pretrain_G_{}.mdl".format(ts)))
    gen.load_state_dict(torch.load(os.path.join(pretrained_gen_path, "pretrain_G.mdl")))
    # gen.load_VAE(load_VAE_path)

    """
    PRETRAIN DISCRIMINATOR
    """
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adam(dis.parameters(), lr = args.dis_lr)
    # train_discriminator(dis, dis_optimizer, datasets, gen, sample_num=50000, ratio_pos = 1.0, ratio_neg = 1.0, d_steps= 3, epochs=20)
    # torch.save(dis.state_dict(), os.path.join(pretrained_dis_path, "pretrain_D_{}.mdl".format(ts)))
    dis.load_state_dict(torch.load(os.path.join(pretrained_dis_path, "pretrain_D.mdl")))

    print('\nStarting Adversarial Training...\n')
    for epoch in range(args.ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ')
        sys.stdout.flush()
        train_generator_PG(gen, dis, datasets, num_samples = 30000, epochs=7)
        if epoch % args.save_epoch == 0: torch.save(gen.state_dict(), os.path.join(pretrained_gen_path, "G_{}.mdl".format(epoch)))

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, datasets, gen, sample_num=50000, ratio_pos=1.1, ratio_neg=1.0, d_steps=3, epochs=10)
        if epoch % args.save_epoch == 0: torch.save(dis.state_dict(), os.path.join(pretrained_dis_path, "D_{}.mdl".format(epoch)))
        print()
        test_g_d(gen, dis, real_data_samples=datasets)

    print("Good Luck !!!")
    print(tracker)
