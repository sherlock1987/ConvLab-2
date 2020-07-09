from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import os
from convlab2.policy.mle.idea4.GAN1.generator import Generator
from convlab2.policy.mle.idea4.GAN1.discriminator import Discriminator
from convlab2.policy.mle.idea4.GAN1.helpers import prepare_data
import pickle
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device != "cuda"

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

TEST = True
CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 10
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
gen_lr = 1e-2

DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_path = os.path.join(root_dir, "mle/processed_data")
load_VAE_path = os.path.join(root_dir, "mle/idea4/model/VAE_39.pol.mdl")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    """
    1. ----Train at each epoch, and val
    2. Test at last epoch.
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

            if len(prev_list) == BATCH_SIZE:
                gen_opt.zero_grad()
                loss = gen.batchNLLLoss(prev_list, bf_temp, target_temp)
                loss.backward()
                gen_opt.step()
                total_loss += loss.data.item()

                # empty the clip
                prev_list = []
                bf_temp = tensor([])
                target_temp = tensor([])

            if ceil((iteration / BATCH_SIZE) % 100) == 0:
                print('.', end='')
                sys.stdout.flush()
        total_loss = total_loss / float(iteration)
        print('average_train_Loss = %.4f' % (total_loss), end="   ")

        # do validation loss over here.
        gen.eval()
        sys.stdout.flush()
        total_loss = 0
        prev_list = []
        bf_temp = tensor([])
        target_temp = tensor([]).to(DEVICE)
        for iteration, ele in enumerate(real_data_samples["val"]):
            # from here to get input, and target.
            prev, bf, target = real_data_samples["val"][iteration]
            prev_list.append(prev.to(DEVICE))
            bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
            target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

            if len(prev_list) == BATCH_SIZE:
                gen_opt.zero_grad()
                loss = gen.batchEVAL(prev_list, bf_temp, target_temp)
                total_loss += loss.data.item()

                # empty the clip
                prev_list = []
                bf_temp = tensor([])
                target_temp = tensor([])

        total_loss = total_loss / float(iteration)
        print('average_val_Loss = %.4f' % (total_loss))

    # test
    gen.eval()
    # set clip
    sys.stdout.flush()
    total_loss = 0
    prev_list = []
    bf_temp = tensor([])
    target_temp = tensor([]).to(DEVICE)
    for iteration, ele in enumerate(real_data_samples["val"]):
        # from here to get input, and target.
        prev, bf, target = real_data_samples["val"][iteration]
        prev_list.append(prev.to(DEVICE))
        bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
        target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

        if len(prev_list) == BATCH_SIZE:
            gen_opt.zero_grad()
            loss = gen.batchEVAL(bf_temp, target_temp)
            total_loss += loss.data.item()

            # empty the clip
            prev_list = []
            bf_temp = tensor([])
            target_temp = tensor([])

    total_loss = total_loss / float(iteration)
    print('average_test_Loss = %.4f' % (total_loss))


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    # sample some stuff.
    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

    print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

# MAIN
if __name__ == '__main__':
    gen    = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu = CUDA)
    dis    = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu = CUDA)
    gen.load_VAE(load_VAE_path)
    print("load VAE model successfully")
    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()
    # DATA READING
    print("Starting Loading Data")
    splits = ['train', 'val']
    # splits = ['val']

    if TEST: splits.append("test")
    datasets = {}
    for split in splits:
        with open(os.path.join(data_path, 'sa_prev_bf_a_{}.pkl'.format(split)), 'rb') as f:
            datasets[split] = pickle.load(f)[split]
    print("Data Loading Successful")
    print("Starting Processing Data into [prev: bf: a]")

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=gen_lr)
    train_generator_MLE(gen, gen_optimizer, datasets, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 50, 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                               start_letter=START_LETTER, gpu=CUDA)
    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 1)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 5, 3)