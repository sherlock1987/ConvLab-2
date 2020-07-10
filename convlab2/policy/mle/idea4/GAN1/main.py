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
import convlab2.policy.mle.idea4.GAN1.helpers
from convlab2.policy.mle.idea4.GAN1.helpers import prepare_data, prepare_discriminator_data, oracle_sample, batchwise_sample
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
BATCH_SIZE_D = 32
MLE_TRAIN_EPOCHS = 10
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
gen_lr = 1e-2
dis_lr = 1e-3

DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64
# path stuff
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_path = os.path.join(root_dir, "mle/processed_data")
load_VAE_path = os.path.join(root_dir, "mle/idea4/model/VAE_39.pol.mdl")
# load_VAE_path = os.path.join(root_dir, "mle/idea4/model/VAE_39_complete.pol.mdl")

pretrained_gen_path = os.path.join(root_dir, "mle/idea4/GAN1/Gen")
pretrained_dis_path = os.path.join(root_dir, "mle/idea4/GAN1/Dis")
if not os.path.exists(pretrained_gen_path): os.makedirs(pretrained_gen_path)
if not os.path.exists(pretrained_dis_path): os.makedirs(pretrained_dis_path)


tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    """
    1. ----Train, val at each epoch.
    2. Test at last epoch.
    """
    # seq_train = ["train", "val", "test"]
    seq_train = ["val", "train", "test"]
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
    for iteration, ele in enumerate(real_data_samples["test"]):
        # from here to get input, and target.
        prev, bf, target = real_data_samples["test"][iteration]
        prev_list.append(prev.to(DEVICE))
        bf_temp = torch.cat((bf_temp, bf.to(DEVICE)), dim=1)
        target_temp = torch.cat((target_temp, target.to(DEVICE)), dim=1)

        if len(prev_list) == BATCH_SIZE:
            gen_opt.zero_grad()
            loss = gen.batchEVAL(prev_list,bf_temp, target_temp)
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

def train_discriminator(discriminator, dis_opt, real_data_samples, generator, sample_num, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # generating a small validation set before training (using oracle and generator)
    """
    1. Shuffle data first
    2. Take the positive to generate negative samples.
    3. Using neg, pos to update D, the thing is, we only have one neg at one time if generator is not update.
    4. If D performs pretty good, we will update G at more time. I think so!!
    5. Note pos_num and neg_num could be totally different. Let's check out in the future.
    """
    # random dataset
    data_cur = real_data_samples["val"]
    random.shuffle(data_cur)
    # repeat this for d_steps
    generator.train()
    for d_step in range(d_steps):
        # do sample first
        # generator.eval()
        sample_oracle_data = oracle_sample(data_cur, sample_num)
        pos_train, neg_trian = generator.sample(sample_oracle_data)
        train_inp, train_target = prepare_discriminator_data(pos_train, neg_trian, gpu=CUDA)
        loss_fn = nn.BCELoss(reduction="sum")
        for epoch in range(epochs):
            discriminator.train()
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0
            # go to the batch
            batch_num = np.ceil(sample_num*2 / BATCH_SIZE_D)
            for i in range(int(batch_num)):
                pointer = i * BATCH_SIZE_D
                if i == int(batch_num-1):
                    # for　the　last batch
                    inp, target = train_inp[0][pointer: sample_num - 1].unsqueeze(0), train_target[pointer: sample_num - 1].unsqueeze(0).unsqueeze(0)
                    # 200 - 6*32 - 1 = 7
                    target = target.view(1, -1, 1)
                else:
                    inp, target = train_inp[0][pointer: pointer + BATCH_SIZE_D].unsqueeze(0), train_target[pointer: pointer + BATCH_SIZE_D].unsqueeze(0).unsqueeze(0)
                    target = target.view(1, -1, 1)

                dis_opt.zero_grad()
                inp = inp.float()
                out = discriminator(inp)
                loss = loss_fn(out, target)
                loss.backward()
                if epoch == 8:
                    pass
                # loss = torch.sum(out - target.view(1,-1,1))
                # for name, param in discriminator.named_parameters():
                #     print(name)
                #     print(param.grad)
                dis_opt.step()

                total_loss += loss.data.item()
                # for balancing data.
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()
                # . . .
                if (i / BATCH_SIZE_D ) % 10 == 0:
                    print('.', end='')
                    sys.stdout.flush()
            total_loss/=  sample_num
            total_acc /=  sample_num

            # do validation
            discriminator.eval()
            sample_oracle_data = oracle_sample(data_cur, 200)
            # based on pos, to get neg. Then get embedding
            pos_val, neg_val = generator.sample(sample_oracle_data)
            val_inp, val_target = prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)
            # do loss again
            val_pred = discriminator(val_inp.float())
            loss_val = loss_fn(val_pred, val_target.view(1, -1, 1))
            loss_val = loss_val.item()
            loss_val /= 200
            val_acc = torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()
            val_acc /= 200
            print(' average_loss = %.4f, train_acc = %.4f, val_loss = %.4f ,val_acc = %.4f' % (total_loss, total_acc, loss_val, val_acc))


# MAIN
if __name__ == '__main__':
    gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu = CUDA)
    dis = Discriminator()
    gen.load_VAE(load_VAE_path)
    print("load VAE model successfully")
    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()
    # DATA READING
    print("Starting Loading Data")
    # splits = ['train', 'val']
    splits = ['val']
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
    # train_generator_MLE(gen, gen_optimizer, datasets, MLE_TRAIN_EPOCHS)
    # load again, clear optimizer
    # torch.save(gen.state_dict(), os.path.join(pretrained_gen_path, "pretrain_G.mdl"))
    gen.load_state_dict(torch.load(os.path.join(pretrained_gen_path, "pretrain_G.mdl")))
    gen.load_VAE(load_VAE_path)
    # train_generator_MLE(gen, gen_optimizer, datasets, MLE_TRAIN_EPOCHS)

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adam(dis.parameters(), lr = dis_lr)
    train_discriminator(dis, dis_optimizer, datasets, gen, sample_num=200, d_steps=5, epochs=30)

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
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, d_steps=5, epochs=3)