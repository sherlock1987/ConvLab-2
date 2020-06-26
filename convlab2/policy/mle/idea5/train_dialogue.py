import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import pickle
from tensorboardX import SummaryWriter
from convlab2.policy.mle.idea5.utils import expierment_name

import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import pickle
from tensorboardX import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"



device = "cuda" if torch.cuda.is_available() else "cpu"
# define the loss function, model_dialogue could also use that.
pos_weights = torch.full([549], 2, dtype=torch.float).to(device)
reconstruction_loss = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
classification_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

def loss_fn(logp, target, disc_res, disc_tar):
    loss1 = reconstruction_loss(logp, target)
    loss2 = classification_loss(disc_res, disc_tar)
    return loss1, loss2


def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    # splits = ['train', 'val'] + (['test'] if args.test else [])
    splits = ['train'] + (['test'] if args.test else [])

    datasets_real = OrderedDict()
    datasets_fake = OrderedDict()
    for split in splits:
        with open(os.path.join("/home/raliegh/图片/ConvLab-2/convlab2/policy/mle/processed_data",
                               'sa_element_{}_real.pkl'.format(split)), 'rb') as f:
            datasets_real[split] = pickle.load(f)
        # with open(os.path.join("/dockerdata/siyao/ft_local/ConvLab/convlab2/policy/mle/multiwoz/processed_data/",
        #                        'sa_element_{}_fake.pkl'.format(split)), 'rb') as f:
        #     datasets_fake[split] = pickle.load(f)

    model = dialogue_VAE(
        max_sequence_length= 60,
        embedding_size= 549,
        rnn_type= "gru",
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    model.to(device)
    # print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

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

    # NLL = torch.nn.NLLLoss(size_average=False)
    # Reconstruction_loss = torch.nn.BCELoss(reduction="sum")

    # def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):
    #
    #     # cut-off unnecessary padding from target, and flatten
    #     # target = target[:, :torch.max(length).item()].contiguous().view(-1)
    #     # logp = logp.view(-1, logp.size(2))
    #
    #     # Negative Log Likelihood
    #     # NLL_loss = NLL(logp, target)
    #     loss = reconstruction_loss(logp, target)
    #
    #     # KL Divergence
    #     KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    #     KL_weight = kl_anneal_function(anneal_function, step, k, x0)
    #
    #     return loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    # start from here.
    batch_id = 0
    for epoch in range(args.epochs):
        for split in splits:

            data_loader_real = datasets_real[split][split]
            # data_loader_fake = datasets_fake[split][split]
            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            temp = []
            discriminator_target = []
            for iteration, batch in enumerate(data_loader_real):
                if batch.size(1) >1:
                    temp.append(batch)
                    a_sys = batch[0][-1][340:549].clone()
                    domain = [0] * 9
                    for i in range(a_sys.shape[0]):
                        if a_sys[i].item() == 1.:
                            if 0 <= i <= 39:
                                domain[0] = 1
                            elif 40 <= i <= 58:
                                domain[8] = 1
                            elif 59 <= i <= 63:
                                domain[1] = 1
                            elif 64 <= i <= 110:
                                domain[2] = 1
                            elif 111 <= i <= 114:
                                domain[3] = 1
                            elif 115 <= i <= 109:
                                domain[4] = 1
                            elif 110 <= i <= 160:
                                domain[5] = 1
                            elif 170 <= i <= 204:
                                domain[6] = 1
                            elif 205 <= i <= 208:
                                domain[7] = 1

                    # temp.append(data_loader_fake[iteration])
                    # discriminator_target.append(1)
                    # discriminator_target.append(0)

                    discriminator_target.append(domain)

                    if (iteration+1) % (args.batch_size) == 0:
                        batch_size = len(temp)
                        # Forward path for VAE
                        original_input, logp, disc_res = model(temp)
                        # original_input, logp, mean, logv, z = model(temp, max_len)

                        # loss calculation
                        loss1, loss2 = loss_fn(logp, original_input.to("cuda"), disc_res, torch.tensor(discriminator_target).float().to("cuda"))
                        loss = loss1 + loss2 * 2
                        if (batch_id+1) % 1000 == 0:
                            print("loss1 & 2:",loss1.item()/batch_size, loss2.item()/batch_size)
                        # NLL_loss, KL_loss, KL_weight = loss_fn(logp, original_input.to("cuda"),
                        #                                        max_len, mean, logv, args.anneal_function, step, args.k, args.x0)
                        #loss = (NLL_loss + KL_weight * KL_loss)

                        # evluation stuff
                        # backward + optimization
                        if split == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            step += 1

                        # bookkeepeing
                        tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.detach().unsqueeze(0)))
                        if split == "test":
                            l1_loss = torch.sum(torch.abs((logp > 0.5).type(torch.FloatTensor) - original_input)).to("cuda")
                            tracker['l1_loss'] = torch.cat((tracker['l1_loss'], l1_loss.unsqueeze(0)))

                        if args.tensorboard_logging and (batch_id+1) % args.print_every == 0:
                            writer.add_scalar("%s/ELBO"%split.upper(), loss.item()/batch_size,  batch_id)
                            # writer.add_scalar("%s/NLL Loss"%split.upper(), NLL_loss.item()/batch_size,  batch_id)
                            # writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.item()/batch_size,  batch_id)
                            # writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, batch_id)

                        # if (batchID+1) % args.print_every == 0: # or iteration+1 == len(data_loader):
                        #     print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        #         %(split.upper(), batchID, len(data_loader)-1, loss.item()/batch_size, NLL_loss.item()/batch_size, KL_loss.item()/batch_size, KL_weight))

                        # if split == 'valid':
                        #     if 'target_sents' not in tracker:
                        #         tracker['target_sents'] = list()
                        #
                        #     tracker['target_sents'] += idx2word(batch['target'].tolist(), i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                        #     tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

                        temp = []
                        discriminator_target = []
                        max_len = 0
                        batch_id += 1

            total_len = 0
            for ele in temp:
                total_len += ele.size(1)
            print("evaluation:  ", (torch.sum(original_input) / total_len).item(),
                  (torch.sum(torch.abs((logp > 0).float() - original_input.to("cuda"))) / total_len).item())

            print("%s Epoch %02d/%i, Mean ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])/args.batch_size ))

            if split == "test":
                print("test L1 loss:  ",(torch.mean(tracker['l1_loss']) / args.batch_size).item())

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO"%split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train' and (epoch+1) % 10 == 0:
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s"%checkpoint_path)

    save_path = "./bin/"
    torch.save(model.state_dict(), save_path + "idea6_domain.pol.mdl")


if __name__ == '__main__':
    # args stuff
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
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

    parser.add_argument('-v', '--print_every', type=int, default=20)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    args_idea5 = parser.parse_args()

    args_idea5.rnn_type = args_idea5.rnn_type.lower()
    args_idea5.anneal_function = args_idea5.anneal_function.lower()

    assert args_idea5.rnn_type in ['rnn', 'lstm', 'gru']
    assert args_idea5.anneal_function in ['logistic', 'linear']
    assert 0 <= args_idea5.word_dropout <= 1
    from utils import expierment_name
    from convlab2.policy.mle.idea5.model_dialogue import dialogue_VAE
    main(args_idea5)
