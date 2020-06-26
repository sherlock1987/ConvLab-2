import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
try:
    from utils import to_var
except Exception as e:
    from .utils import to_var

import torch.nn.functional as F
import torch.tensor as tensor
import copy
import numpy as np
from collections import defaultdict
from convlab2.policy.mle.idea5.train_dialogue import loss_fn

class Compare():
    def __init__(self):
        pass

    def projection(self, input, oracle):
        # one vecter on the oracle vector
        cos = self.cos_sam(input, oracle)
        length_oracle = torch.sqrt(torch.sum(oracle * oracle))
        length_vec = torch.sqrt(torch.sum(input * input))
        output = length_vec * cos/length_oracle
        return output

    def cos_sam(self, input, oracle):
        # print(input.shape, oracle.shape)
        upper = torch.sum(input * oracle)
        lower_1 = torch.sqrt(torch.sum(input * input))
        lower_2 = torch.sqrt(torch.sum(oracle* oracle))
        return (upper/(lower_1*lower_2))

    def distance(self, input, oracle):
        diff = input - oracle
        output = torch.sqrt(torch.sum(diff * diff))
        return output

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
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.discriminator_layer1 = nn.Linear(hidden_size * self.hidden_factor + 340, 128)
        self.discriminator_layer2 = nn.Linear(128,32)
        self.discriminator_layer3 = nn.Linear(32,9)
        self.bceLoss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr= 0.001)

    def mask_last_action(self, input):
        output = copy.deepcopy(input)
        for s in output:
            s[0, -1, 340:549] = 0.
        return output

    def get_last_action(self, input):
        return input[0][-1][340:549]

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
            bf = torch.tensor([]).to("cuda") # empty clips
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

    def forward(self, input_sequence):
        """
        using the previous to do AE, and use prev hidden + bf to make domain classifier
        data.size(1) >= 2
        :param input_sequence:
        :param length:
        :return: loss
        """
        batch_size = len(input_sequence)
        # extract the previous fea, and current bf
        prev, bf = self.extract_prev_bf(input_sequence)
        # remember to max_len - 1
        max_len = self.get_max_len(input_sequence)
        original_input_tensor, padded_input_sequence, sorted_lengths, sorted_idx = self.concatenate_zero(prev, max_len - 1)
        padded_input_sequence_decoder = padded_input_sequence.clone()
        padded_input_sequence = self.linear2(self.relu(self.linear1(padded_input_sequence.to("cuda"))))

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
        # # z = z * std + mean
        # # DECODER latent to real hidden states
        # hidden = self.latent2hidden(z)

        # from [hidden : belief state] to make classify
        hidden_bf = torch.cat((hidden, bf), 1)
        disc_res = self.discriminator_layer3(self.relu(self.discriminator_layer2(self.relu(self.discriminator_layer1(hidden_bf)))))

        # add VAE is better than NONE
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        # why dropout this one.
        # In fact, it is going to make this stuff work.

        # decoder forward pass, hidden [ , , ] three dimention
        # padded_input_sequence_decoder = torch.cat((torch.zeros(batch_size,1,self.input_size),padded_input_sequence_decoder),dim=1)
        # original_input_tensor = torch.cat((original_input_tensor,torch.zeros(batch_size,1,self.input_size)),dim=1)
        padded_input_sequence_decoder = self.linear2(self.relu(self.linear1(padded_input_sequence_decoder.to("cuda"))))
        input_dropout = self.dropout(padded_input_sequence_decoder)

        # pack stuff
        packed_input = rnn_utils.pack_padded_sequence(input_dropout, sorted_lengths.data.tolist(), batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden)
        # output layer first!
        # unpack stuff
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True,padding_value=0.0)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        # b,s,_ = padded_outputs.size()

        outputs_layer = self.output_layer(padded_outputs)

        # z is the distribution.
        # outputs = self.output_layer(padded_outputs)
        #return original_input_tensor, outputs_layer, mean, logv, z
        disc_res = disc_res[reversed_idx]
        return original_input_tensor, outputs_layer, disc_res

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

        input = self.linear2(self.relu(self.linear1(input.to("cuda"))))

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

    def get_reward(self, r, s, a, mask,  globa_bool = False, global_type = "mask"):
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
        data_sub = []
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
                # for the last one, the reward should follow the system. 5, 40, -1, that's it.
                last_reward = r[i].item()
                reward_collc.append(last_reward)
                if globa_bool:
                    global_score = self.get_score_global(input, global_type = global_type)
                    global_score.append(0)
                    # add global score
                    for i in range(len(reward_collc)):
                        reward_collc[i] += global_score[i]
                #
                data_sub.append(input)
                data_collc[sum(reward_collc)].append(data_sub)
                data_reward_collc[sum(reward_collc)].append(reward_collc)

                reward_predict += reward_collc
                # clear
                s_temp = torch.tensor([])
                a_temp = torch.tensor([])
                reward_collc = []
                data_sub = []
            else:
                # to describe whether it is the first, first just give 1.
                if input.size(1) > 1:
                    domain_score = self.get_score_domain(input)
                    reward_collc.append(domain_score.item())
                    data_sub.append(input)
                else:
                    reward_collc.append(0.5)
        reward_predict = torch.tensor(reward_predict)
        # update the reward model first.
        self.update(data_collc, 3, 32)
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
            if num_dia > 0:
                if ele > 40:
                    batch+=data[ele]
                    num_dia -=1
        input = []
        discriminator_target = []
        for i in range(len(batch)):
            temp = copy.deepcopy(batch[i])
            for j in range(len(temp)):
                input.append(temp[j])
                one = temp[j].clone()[0][-1][340:549]
                discriminator_target.append(self.domain_classifier(one))
                if len(input) > 10:
                    original_input, logp, disc_res = self.forward(input)
                    loss1, loss2 = loss_fn(logp, original_input.to("cuda"), disc_res,
                                           torch.tensor(discriminator_target).float().to("cuda"))
                    loss = loss1 + loss2 * 2
                    print("LOSS(updating for reward model): ",loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # clear stuff
                    input = []
                    discriminator_target = []

    def get_reward_global(self, r, s, a, mask, globa_type = "mask"):
        """
        Only consist the global score.
        :param r: reward, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :param s: state, Tensor, [b,340]
        :param a: action, Tensor, [b,209]
        """
        reward_predict = []
        batchsz = r.shape[0]
        s_temp = torch.tensor([])
        a_temp = torch.tensor([])
        # stor the data elementise
        reward_collc = []
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
                # for the last one, the reward should follow the system. 5, 40, -1, that's it.
                last_reward = r[i].item()
                global_score = self.get_score_global(input, global_type= globa_type)
                global_score.append(last_reward)
                # add global score
                s_temp = torch.tensor([])
                a_temp = torch.tensor([])
                reward_predict += global_score
            #   compute the last one, terminate clear the button, that is okay for us.
            else:
                pass

        reward_predict = torch.tensor(reward_predict)

        return reward_predict
    # sub func for reward computation
    def get_score_domain(self, input):
        """
        Get current dialogue score for the domain.
        This is a sub_function
        :param input: [ , , ]
        :param gloabl: bool, true meaning to this stuff.
        :param gloabl type: Hindsight, mask
        :return: reward computation
        """
        output_action = self.get_last_action(input.clone())
        input_rnn, bf = self.extract_prev_bf(input)
        batch_size = input_rnn.shape[0]
        # ENCODER
        input_mask = self.linear2(self.relu(self.linear1(input_rnn.to("cuda"))))
        _, hidden = self.encoder_rnn(input_mask)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
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
        # # z = z * std + mean
        # # DECODER latent to real hidden states
        # hidden = self.latent2hidden(z)

        # discriminate
        hidden_bf = torch.cat((hidden, bf), 1)
        disc_res = self.discriminator_layer3(self.relu(self.discriminator_layer2(self.relu(self.discriminator_layer1(hidden_bf)))))
        action_domain = self.domain_classifier(output_action)
        prob = self.sigmoid(disc_res)
        score = torch.sum(torch.tensor(action_domain).to("cuda") * prob.squeeze(0))
        return score
    # sub func for reward computation
    def get_score_fake(self, input):
        """
        Get current dialogue score for the fake/real.
        This is a sub_function
        This is only for descriminating fake or real data.
        :param input_sequence: [ , , ]
        :param gloabl: bool, true meaning to this stuff.
        :param gloabl type: Hindsight, mask
        :return: reward computation
        """
        # pack stuff and unpack stuff later.
        batch_size = input.size(0)
        input_mask = self.mask_last_action(input)
        input_sequence = self.linear2(self.relu(self.linear1(input.to("cuda"))))
        _, hidden = self.encoder_rnn(input_sequence)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
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
        # # z = z * std + mean
        # # DECODER latent to real hidden states
        # hidden = self.latent2hidden(z)

        # discriminate
        disc_res = self.discriminator_layer3(self.relu(self.discriminator_layer2(self.relu(self.discriminator_layer1(hidden)))))
        true_prob = F.softmax(disc_res.squeeze())[1] - 0.5

        return true_prob

    # sub func for reward computation
    def get_score_global(self, input, global_type = "cos"):
        """
        :param input:[ , , ]
        :param global_type: "cos", "mask"
        :return: list
        """
        res = []
        reward_ori = []
        if global_type == "cos":
            input_sequence = self.linear2(self.relu(self.linear1(input.to("cuda"))))
            whole, hidden = self.encoder_rnn(input_sequence)
            length = len(whole[0])
            for i in range(length-1):
                curr_h = whole[0][i].unsqueeze(0)
                reward_cur = self.compare.cos_sam(curr_h, hidden).item()
                res.append(reward_cur)
            prev = res.copy()
            for i in reversed(range(1,length-1)):
                res[i] = res[i] - res[i-1]

        elif global_type == "mask":
            length = len(input[0])
            input_sequence = self.linear2(self.relu(self.linear1(input.to("cuda"))))
            whole, hidden_oracle = self.encoder_rnn(input_sequence)

            zero = torch.zeros(209)
            for i in range(length - 1):
                input_linear = input.clone()
                input_linear[0][i][340:549] = zero
                input_sequence = self.linear2(self.relu(self.linear1(input_linear.to("cuda"))))
                whole, hidden = self.encoder_rnn(input_sequence)
                reward_cur = (1 - self.compare.cos_sam(hidden, hidden_oracle).item())*10 - 0.5
                res.append(reward_cur)
                reward_ori.append(self.compare.cos_sam(hidden, hidden_oracle).item())
        else:
            raise NotImplementedError
        return res


