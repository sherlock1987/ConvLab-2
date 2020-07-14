import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import pickle
from convlab2.policy.mle.idea4.utils import expierment_name
from convlab2.policy.mle.idea4.model_dialogue import dialogue_VAE
import random
import sys
# set name stuff.
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
data_path = os.path.join(root_dir, "policy/mle/processed_data")
save_path = os.path.join(root_dir,"policy/mle/idea4/model")
if not os.path.exists(save_path): os.makedirs(save_path)
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
"""
Training 
1.1 Set seed first. ALl of training for the model is seed = 1
1.2. VAE stuff, to pretrain on the 123 data or 11 data. And test it.
1.3 train and val, last is test.
    train
    val
        untill val is going higher for 2 epoches, takes the best one
        test
2. Training on D and G, using VAE.
"""

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    splits = ['train', 'val']
    splits = splits + (['test'] if args.test else [])
    datasets = OrderedDict()
    # reading data.
    for split in splits:
        # with open(os.path.join(data_path, 'sa_{}.pkl'.format(split)), 'rb') as f:
        with open(os.path.join(data_path, 'sa_element_{}_real.pkl'.format(split)), 'rb') as f:
            datasets[split] = pickle.load(f)[split]

    model = dialogue_VAE(
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )
    # check if cuda is working.
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("cuda is not available at all!!!")
        raise Exception

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    def kl_anneal_function(anneal_function, step, k, x0):
        """
        :param anneal_function:
        :param step:
        :param k:
        :param x0:
        :return:
        """
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)
    # define loss func
    pos_weights = torch.full([549],2).to(device)
    Reconstruction_loss = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #Reconstruction_loss = FocalLoss(549)
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        # target = target[:, :torch.max(length).item()].contiguous().view(-1)
        # logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        # NLL_loss = NLL(logp, target)
        loss = Reconstruction_loss(logp, target)

        # KL Divergence, in a inverse way, the lower, the better.
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)
        # KL_loss = 0.
        # KL_weight = 0.
        return loss, KL_loss, KL_weight

    def test():
        # runing for test.
        batchID = 0
        data_loader = datasets['test']
        # name a tracker for writing logs.
        tracker_test = defaultdict(tensor)
        model.eval()
        temp = []
        max_len = 0
        # start loop
        for iteration, element in enumerate(data_loader):
            temp.append(element.to(device))
            max_len = max(max_len, element.size(1))
            if (iteration + 1) % args.batch_size == 0:
                batch_size = len(temp)
                input, logp, mean, logv, z = model(temp)
                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, input.to("cuda"), max_len, mean, logv, args.anneal_function, step, args.k, args.x0)
                loss = (NLL_loss + KL_weight * KL_loss)
                # recording for everything.
                tracker_test['ELBO'] = torch.cat((tracker_test['ELBO'], loss.detach().unsqueeze(0)))
                # do evaluation in val
                test_loss = torch.sum(torch.abs((logp > 0.5).type(torch.FloatTensor) - input)).to("cuda")
                tracker_test['test_diff'] = torch.cat((tracker_test['test_diff'], test_loss.unsqueeze(0)))
                # empty the clip
                temp = []
                max_len = 0
                batchID += 1
        loss_test = (torch.mean(tracker_test['test_diff']) / args.batch_size).item()
        print("Test loss:  ", loss_test)

    step = 0
    # start from here.
    batchID = 0
    for epoch in range(args.epochs):
        # [train, val, test]
        loss_val = 100000
        model_list = []
        for split in splits:
            data_loader = datasets
            # name a tracker for writing logs.
            tracker = defaultdict(tensor)
            if split == 'train':
                model.train()
            # val, test will happen in
            else:
                model.eval()
            temp = []
            max_len = 0
            # start loop
            for iteration, element in enumerate(data_loader[split]):
                temp.append(element.to(device))
                max_len = max(max_len, element.size(1))
                if (iteration + 1) % args.batch_size == 0:
                    batch_size = len(temp)
                    input, logp, mean, logv, z = model(temp)

                    # loss calculation
                    NLL_loss, KL_loss, KL_weight = loss_fn(logp, input.to("cuda"), max_len, mean, logv, args.anneal_function, step, args.k, args.x0)
                    loss = (NLL_loss + KL_weight * KL_loss)

                    # backward + optimization
                    if split == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        step += 1

                    # recording for everything.
                    tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.detach().unsqueeze(0)))
                    # do evaluation in val.
                    if split == "val":
                        # record evaluation
                        test_loss = torch.sum(torch.abs((torch.sigmoid(logp) > 0.5).type(torch.FloatTensor) - input)).to("cuda")
                        tracker['test_diff'] = torch.cat((tracker['test_diff'], test_loss.unsqueeze(0)))

                    if args.tensorboard_logging and (batchID + 1) % args.print_every == 0:
                        # write loss after each training batch
                        writer.add_scalar("%s/ELBO"%split.upper(), loss.item()/batch_size,  batchID)
                        writer.add_scalar("%s/NLL Loss"%split.upper(), NLL_loss.item()/batch_size,  batchID)
                        writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.item()/batch_size,  batchID)
                        writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, batchID)

                    if (batchID+1) % args.print_every == 0: # or iteration+1 == len(data_loader):
                        print("%s Batch %04d/%i, Loss %9.4f, RC-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                            %(split.upper(), batchID, len(data_loader)-1, loss.item()/batch_size, NLL_loss.item()/batch_size, KL_loss.item()/batch_size, KL_weight))
                    # empty the clip
                    temp = []
                    max_len = 0
                    batchID += 1

            # print only after one epoch finished, only for training.
            if split == "train": print("%s epoch %02d/%i, Inverse Mean ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])/args.batch_size ))

            if split == "val":
                loss_val_curr = (torch.mean(tracker['test_diff']) / args.batch_size).item()
                print("Val evaluation loss:  ", loss_val_curr)
                cur_min = min(loss_val, loss_val_curr)
                # upper time.
                if loss_val_curr > cur_min:
                    model_list.append(model.state_dict())
                    if len(model_list) == args.upper_time:
                        # store first model and do test, and then print this stuff.
                        checkpoint_path = os.path.join(save_path, "VAE_{}_{}.mdl".format(epoch, ts))
                        best_model = model_list[0]
                        torch.save(best_model, checkpoint_path)
                        print("Model saved at %s" % checkpoint_path)
                        # do test
                        if args.test:
                            test()
                            exit()
                            # end code
                else:
                    # go downer
                    loss_val = cur_min

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO"%split.upper(), torch.mean(tracker['ELBO']), epoch)

            # # save checkpoint
            # if split == 'train' and (epoch + 1) % 10 == 0:
            #     os.makedirs(save_path)
            #     checkpoint_path = os.path.join(save_path, "VAE%i.mdl"%(epoch))
            #     torch.save(model.state_dict(), checkpoint_path)
            #     print("Model saved at %s" %checkpoint_path)
    test()
    print("model save at {}".format(save_path))
    torch.save(model.state_dict(), save_path + "/VAE_{}_123.pol.mdl".format(epoch+1))


if __name__ == '__main__':
    # args stuff
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', type = bool, default=True)

    parser.add_argument('-ep', '--epochs', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=549)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True)
    parser.add_argument('-ls', '--latent_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=1)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=1)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=100)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')
    parser.add_argument('-up', '--upper_time', type=int, default=3)

    args = parser.parse_args()
    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1
    # start runing code. Training a VAE.
    main(args)
