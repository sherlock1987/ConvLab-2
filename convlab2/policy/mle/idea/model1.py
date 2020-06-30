import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
try:
    from utils import to_var
except Exception:
    from .utils import to_var

import torch.nn.functional as F
import copy
from collections import defaultdict

# define the loss function, model_dialogue could also use that.
pos_weights = torch.full([549], 2, dtype=torch.float).to("cuda")
reconstruction_loss = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
classification_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

def loss_fn(logp, target, disc_res, disc_tar):
    loss1 = reconstruction_loss(logp, target)
    #loss2 = classification_loss(disc_res, disc_tar)
    return loss1#, loss2, torch.mean(torch.abs(disc_tar - disc_res))

class Compare():
    def __init__(self):
        pass

    def cos_sam(self, input, oracle):
        # print(input.shape, oracle.shape)
        upper = torch.sum(input * oracle)
        lower_1 = torch.sqrt(torch.sum(input * input))
        lower_2 = torch.sqrt(torch.sum(oracle* oracle))
        return upper/(lower_1*lower_2)


class dialogue_VAE(nn.Module):
    def __init__(self, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.compare = Compare()
        self.max_sequence_length = max_sequence_length
        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_dropout_rate = word_dropout
        # self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.dropout = nn.Dropout(p=0.5)
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM

        else:
            raise ValueError()
        self.input_size = embedding_size
        self.linear1 = nn.Linear(embedding_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.encoder_rnn = rnn(hidden_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(hidden_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.output_layer = nn.Linear(hidden_size * (2 if bidirectional else 1), embedding_size)

        self.bceLoss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.parameters(), lr= 0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mask_last_action(self, input):
        output = copy.deepcopy(input)
        for s in output:
            s[0, -1, 340:549] = 0.
        return output

    def get_last_action(self, input):
        return input[0][-1][340:549].clone()

    def domain_classifier(self, action):
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

    def extract_prev_bf(self, input):
        """
        :param input: list [ [] [] [] [] [] [] [] [] ] or tensor [ , , ]
        :return: prev: list, bf [ 1, 53, 340]
        """
        type_1 = type(input)
        if type(input) == list:
            # this is for forward func
            prev = copy.deepcopy(input)
            output_prev = []
            bf = torch.tensor([]).to(self.device) # empty clips
            for i, ele in enumerate(prev):
                bf_ele = ele[0][-1][0:340].clone().unsqueeze(0)
                ele_prev = ele[0][:-1].unsqueeze(0)
                output_prev.append(ele_prev)
                bf = torch.cat((bf, bf_ele), 0)
        elif type(input) == torch.Tensor:
            # this is for compute
            prev = input.clone()
            bf = prev[0][-1][0:340].clone().unsqueeze(0)
            output_prev = prev[0][:-1].unsqueeze(0)

        else:
            raise NotImplementedError

        return  output_prev, bf

    def get_max_len(self, input_seq):
        """
        :param input_seq: [ ,  , ]
        :return: int
        """
        max_len = 0
        process = copy.deepcopy(input_seq)
        for i in range(len(process)):
            cur_len = process[i].size(1)
            max_len = max(max_len, cur_len)
        return max_len

    def data_mask(self, input):
        """
        :param input: [ , , ] one tensor
        :return: list of tensors, and it should mask one turn, list_len = len([ , , ])
        """
        dia_len = input.size(1)
        input_tensor = input.clone().detach()
        domain_list = []
        mask_input_tensor = []
        mask_id = []
        bf_list = []

        for i in range(dia_len):
            action = input_tensor[0][i][340:549]
            domain_list.append(self.domain_classifier(action))
            bf = input_tensor[0][i][:340].clone()
            bf_list.append(bf)
            one = input_tensor.clone()
            one[0][i][:] = torch.zeros(size = [549])
            mask_input_tensor.append(one)
            mask_id.append(i)

        return mask_input_tensor, mask_id, domain_list, bf_list

    def forward(self, input_sequence, mask_id, input_belief):
        """
        First mask the input_sequence, in train.py.
        data.size(1) >= 2
        :param input_sequence: list of tensors
        :param mask_id:
        :return: loss_1(RC), loss_2(disc)
        """

        batch_size = len(input_sequence)
        max_len = self.get_max_len(input_sequence)

        original_input_tensor, padded_input_sequence, sorted_lengths, sorted_idx = self.concatenate_zero(input_sequence, max_len)
        padded_input_sequence_decoder = padded_input_sequence.clone()
        padded_input_sequence = self.linear2(F.relu(self.linear1(padded_input_sequence.to(self.device))))
        packed_input = rnn_utils.pack_padded_sequence(padded_input_sequence, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state to [b, 1024]
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # # REPARAMETERIZATION
        # # related to latent size, which is 16 (16/256)
        # mean = self.hidden2mean(hidden)
        # logv = self.hidden2logv(hidden)
        # std = torch.exp(0.5 * logv)
        #
        # z = to_var(torch.randn([batch_size, self.latent_size]))
        # z = z * std + mean
        # # DECODER latent to real hidden states
        # hidden = self.latent2hidden(z)
        # # add VAE is better than NONE


        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder forward pass, hidden [ , , ] three dimention
        padded_input_sequence_decoder = self.linear2(F.relu(self.linear1(padded_input_sequence_decoder.to(self.device))))
        input_dropout = self.dropout(padded_input_sequence_decoder)

        # pack stuff
        packed_input = rnn_utils.pack_padded_sequence(input_dropout, sorted_lengths.data.tolist(), batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden)
        # output layer first!
        # unpack stuff
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True,padding_value=0.0)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        # hidden_states
        padded_outputs = padded_outputs[reversed_idx]
        reconstructed_dialogue = self.output_layer(padded_outputs)

        return original_input_tensor, reconstructed_dialogue

    def concatenate_zero(self, input, max_len):
        """
        :param input: list [[,,] [,,] [,,] [,,] [,,] ]
        :param max_len:
        :return:pad input, pad_input_sort, sorted_length, sorted_idx
        """
        output = torch.zeros(size = (len(input), max_len, self.input_size))

        len_list = []
        for i, element in enumerate(input):
            len_dialogue = element.size(1)
            assert len_dialogue <= max_len
            len_list.append(len_dialogue)

            for j in range(len_dialogue):
                output[i][j] = element[0][j]

        pad_input_tensor = output
        sorted_lengths, sorted_idx = torch.sort(torch.tensor(len_list), descending=True)
        # sort stuff
        input_sequence = output.clone()[sorted_idx]
        return pad_input_tensor, input_sequence, sorted_lengths, sorted_idx

    def compress(self,input):

        input = self.linear2(F.relu(self.linear1(input.to(self.device))))

        _, hidden = self.encoder_rnn(input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(1, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        # related to latent size, which is 16 (16/256)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([1, self.latent_size]))
        distribution = z * std + mean

        return mean, logv, distribution, std

    def get_reward(self, r, s, a, mask, needs_update):
        """
        :param r: reward, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :param s: state, Tensor, [b,340]
        :param a: action, Tensor, [b,209]
        """
        reward_predict = []
        batchsz = r.shape[0]
        s_temp = torch.tensor([])
        a_temp = torch.tensor([])
        # store the data elementise
        reward_collc = []
        # store the data for updating
        data_collc = defaultdict(list)
        data_reward_collc = defaultdict(list)
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
            if int(mask[i]) == 0:
                last_reward = r[i].item()

                global_score = self.get_score_global(input, last_reward)

                # add global score
                reward_collc += global_score
                data_collc[last_reward].append(input)
                data_reward_collc[sum(reward_collc)].append(reward_collc)

                reward_predict += reward_collc
                # clear
                s_temp = torch.tensor([])
                a_temp = torch.tensor([])
                reward_collc = []

        reward_predict = torch.tensor(reward_predict)
        print(reward_predict[:10])
        # update the reward model
        if needs_update:
            self.update(data_collc, 16, 32)
        return reward_predict

    def update(self, input, num_dia, bs):
        """
        :param input: dict {num}: [tensors, tensors, tensors]...
        :return:
        """
        data = copy.deepcopy(input)
        # pick the top, then update this stuff.
        key = list(input.keys())
        key.sort(reverse = True)
        batch = []
        for ele in key:
            if ele < 0 :
                break
            for d in data[ele]:
                batch.append(d)
                num_dia -= 1
                if num_dia == 0:
                    break
            if num_dia == 0:
                break

        mask_input_collc = []
        mask_id_collc = []
        domain_collc = []
        bf_collc = []
        for i in range(len(batch)):
            temp = copy.deepcopy(batch[i]) # 1 * 10 * 549
            mask_input_tensor, mask_id, domain_list, input_belief = self.data_mask(temp)
            mask_input_collc += mask_input_tensor
            mask_id_collc += mask_id
            domain_collc = domain_list
            bf_collc = input_belief

        mask_input_batch = []
        mask_id_batch = []
        domain_batch = []
        bf_batch = []
        index = [i for i in range(len(domain_collc))]
        for mask_input_one, domain, mask, bf, iteration in zip(mask_input_collc, domain_collc, mask_id_collc, bf_collc, index):
            mask_input_batch.append(mask_input_one)
            domain_batch.append(domain)
            mask_id_batch.append(mask)
            bf_batch.append(bf)
            if (iteration+1) % bs == 0:

                original_input, logp = self.forward(mask_input_batch,mask_id_batch,bf_batch)
                loss = loss_fn(logp, original_input.to(self.device), 0, 0)
                #loss = loss1 + loss2
                #print("LOSS(updating for reward model): ",loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # clear stuff
                mask_input_batch = []
                mask_id_batch = []
                domain_batch = []
                bf_batch = []

    def get_score_global(self, input, final_r):
        """
        :param input:[ , , ]
        :param global_type: "cos", "mask"
        :return: list
        """
        res = []

        length = len(input[0])
        input_sequence = self.linear2(F.relu(self.linear1(input.to(self.device))))
        whole, hidden_oracle = self.encoder_rnn(input_sequence)

        zero = torch.zeros(549)
        for i in range(length):
            input_linear = input.clone()
            input_linear[0][i][:] = zero
            input_sequence = self.linear2(F.relu(self.linear1(input_linear.to(self.device))))
            whole, hidden = self.encoder_rnn(input_sequence)
            #reward_cur = (1. - self.compare.cos_sam(hidden, hidden_oracle).item())*10 -0.5
            reward_cur = 1. - self.compare.cos_sam(hidden, hidden_oracle).item()
            res.append(reward_cur)
            #reward_ori.append(self.compare.cos_sam(hidden, hidden_oracle).item())

        # length = len(res)
        # res = F.softmax(torch.tensor(res)) * final_r * length
        # for i in range(length):
        #     res[i] -= length-i-1
        res = F.softmax(torch.tensor(res)) * len(res) - 1

        res = res.tolist()

        return res


