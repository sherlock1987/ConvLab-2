import os
import json
import time
import torch
import argparse
import numpy as np
from collections import OrderedDict, defaultdict
import pickle
from tensorboardX import SummaryWriter
from convlab2.policy.mle.idea.model1 import dialogue_VAE as Model1
from convlab2.policy.mle.idea.model1 import loss_fn as loss_fn1
from convlab2.policy.mle.idea.model2 import dialogue_VAE as Model2
from convlab2.policy.mle.idea.model2 import loss_fn as loss_fn2
from convlab2.policy.mle.idea.utils import expierment_name
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def domain_classifier(action):
    """
    :param action: action is from tensor or list of tensor
    :param temp:
    :return:
    """
    if type(action) == torch.Tensor:
        domain = [0] * 9
        a = action.clone()
        for i in range(a.shape[0]):
            if a[i].item() == 1.:
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
    elif type(action) == list:
        pass

    return domain

def data_mask(input):
    """
    :param input: [ , , ] one tensor
    :return: list of tensors, and it should mask one turn, list_len = len([ , , ])
    """
    dia_len = input.size(1)
    input_tensor = input.clone().detach()
    domain_list = []
    prev_input_tensor = []
    mask_input_tensor = []
    mask_id = []
    bf_list = []

    for i in range(dia_len):
        action = input_tensor[0][i][340:549]
        domain_list.append(domain_classifier(action))
        bf = input_tensor[0][i][:340].clone()
        bf_list.append(bf)
        one = input_tensor.clone()
        one[0][i][:] = torch.zeros(size=[549])
        mask_input_tensor.append(one)
        prev_input_tensor.append(input_tensor[0][:i][:].unsqueeze(0).clone())
        mask_id.append(i)

    return mask_input_tensor, mask_id, domain_list, bf_list, prev_input_tensor

def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    # splits = ['train', 'val'] + (['test'] if args.test else [])
    splits = ['train'] + (['test'] if args.test else [])
    # splits = ["test"]
    datasets_real = OrderedDict()
    for split in splits:
        with open(os.path.join("/dockerdata/siyao/ft_local/ConvLab/convlab2/policy/mle/multiwoz/processed_data",
                               'sa_{}.pkl'.format(split)), 'rb') as f:
            datasets_real[split] = pickle.load(f)

    model1 = Model1(
        max_sequence_length= 40,
        embedding_size= 549,
        rnn_type= "gru",
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )
    model2 = Model2()

    model1.to(device)
    model2.to(device)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args,ts)))
        writer.add_text("model1", str(model1))
        writer.add_text("model2", str(model2))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0

    # start from here.
    batch_id = 0
    # data
    processed_data = {}
    for split in splits:
        data_loader_real = datasets_real[split][split]
        data_collc = []
        domain_collc = []
        mask_id_collc = []
        bf_collc = []
        pre_data_collc = []
        for i, batch in enumerate(data_loader_real):
            one_1, one_2, one_3, one_4, one_5 = data_mask(batch)
            data_collc += one_1
            mask_id_collc += one_2
            domain_collc += one_3
            bf_collc += one_4
            pre_data_collc += one_5
        index = [i for i in range(len(domain_collc))]
        processed_data[split] = (data_collc, domain_collc, mask_id_collc, bf_collc, pre_data_collc, index)

    for epoch in range(args.epochs):
        for split in splits:

            tracker = defaultdict(tensor)
            # Enable/Disable Dropout
            if split == 'train':
                model1.train()
                model2.train()
            else:
                model1.eval()
                model2.eval()

            temp = []
            discriminator_target = []
            mask_list = []
            bf_list = []
            data_collc, domain_collc, mask_id_collc, bf_collc, index = processed_data[split]
            for batch, domain, mask, bf, iteration in zip(data_collc, domain_collc, mask_id_collc, bf_collc, index):
                if batch.size(1) >1:
                    temp.append(batch)
                    discriminator_target.append(domain)
                    mask_list.append(mask)
                    bf_list.append(bf)

                    if (iteration+1) % (args.batch_size) == 0:
                        batch_size = len(temp)
                        # Forward path for VAE
                        prediction = model(temp,  torch.stack(bf_list).to("cuda"))
                        # original_input, logp, mean, logv, z = model(temp, max_len)

                        # loss calculation
                        loss, loss2 = loss_fn(prediction, torch.tensor(discriminator_target).float().to("cuda"))
                        #loss = loss1 + loss2
                        if (batch_id+1) % 500 == 0:
                            print("loss1 & 2:",loss.item()/batch_size, loss2.item()/batch_size)

                        # evluation stuff
                        # backward + optimization
                        if split == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            step += 1

                        # bookkeepeing
                        tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.detach().unsqueeze(0)))
                        # if split == "test":
                        #     l1_loss = torch.sum(torch.abs((logp > 0.5).type(torch.FloatTensor) - original_input)).to("cuda")
                        #     tracker['l1_loss'] = torch.cat((tracker['l1_loss'], l1_loss.unsqueeze(0)))

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
                        mask_list = []
                        bf_list = []
                        batch_id += 1

            # print("No. of pos in data:", (torch.sum(original_input) /args.batch_size).item(), "No. of incorrect prediction:",
            #       (torch.sum(torch.abs((logp > 0).float() - original_input.to("cuda"))) /args.batch_size).item())

            print("%s Epoch %02d/%i, total Loss %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])/args.batch_size ))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train' and (epoch+1) % 5 == 0:
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s"%checkpoint_path)

    save_path = "./bin/"
    torch.save(model.state_dict(), save_path + "idea8.pol.mdl")


if __name__ == '__main__':
    # args stuff
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=20)
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

    main(args_idea5)
