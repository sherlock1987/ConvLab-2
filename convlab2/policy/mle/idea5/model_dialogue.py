import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from .utils import to_var
import torch.nn.functional as F
import torch.tensor as tensor

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
        self.discriminator_layer1 = nn.Linear(hidden_size * self.hidden_factor, 128)
        self.discriminator_layer2 = nn.Linear(128,32)
        self.discriminator_layer3 = nn.Linear(32,2)


    def forward(self, input_sequence, max_len):
        """
        :param input_sequence:
        :param length:
        :return:
        """
        # order
        batch_size = len(input_sequence)
        # ENCODER
        # pack stuff and unpack stuff later.
        original_input_tensor, padded_input_sequence, sorted_lengths, sorted_idx = self.concatenate_zero(input_sequence, max_len)
        padded_input_sequence_decoder = padded_input_sequence.clone()
        padded_input_sequence = self.linear2(self.relu(self.linear1(padded_input_sequence.to("cuda"))))

        packed_input = rnn_utils.pack_padded_sequence(padded_input_sequence, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

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
        output = torch.zeros(size = (len(input), max_len, self.input_size))

        len_list = []
        for i, element in enumerate(input):
            len_dialogue = element.size(1)
            len_list.append(len_dialogue)
            left = output[i][1]
            right = element[0]
            for j in range(len_dialogue):
                output[i][j] = element[0][j]

        original_input_tensor = output
        sorted_lengths, sorted_idx = torch.sort(torch.tensor(len_list), descending=True)
        input_sequence = output[sorted_idx]
        return original_input_tensor, input_sequence, sorted_lengths, sorted_idx

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

    def get_reward(self, r, s, a, mask, local=True, globa = True):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
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
                reward_collc.append(last_reward)
                global_score = self.get_score_global(input, global_type= "mask")
                global_score.append(0)
                # add global score
                for i in range(len(reward_collc)):
                    reward_collc[i] += global_score[i]
                s_temp = torch.tensor([])
                a_temp = torch.tensor([])
                reward_predict += reward_collc
                reward_collc = []
            #   compute the last one, terminate clear the button, that is okay for us.
            else:
                with torch.no_grad():
                    # self.g
                    fake_score = self.get_score(input)
                    reward_collc.append(fake_score.item())

        reward_predict = torch.tensor(reward_predict)

        return reward_predict

    def get_score(self, input_sequence):
        """
        :param input_sequence: list
        :param gloabl: bool, true meaning to this stuff.
        :param gloabl type: Hindsight, mask
        :return: reward computation
        """
        # order
        seq_len = len(input_sequence)
        batch_size = input_sequence[0][0]
        input_last = input_sequence[-1]
        # ENCODER
        # pack stuff and unpack stuff later.
        input_sequence = self.linear2(self.relu(self.linear1(input_sequence.to("cuda"))))
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

    def get_score_global(self, input, global_type = "cos"):
        """
        :param input:[ , , ]
        :param global_type: "cos", "mask"
        :return: list
        """
        res = []
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

            zero = torch.zeros(549)

            for i in range(length - 1):
                input_linear = input.clone()
                input_linear[0][i][:] = zero
                input_sequence = self.linear2(self.relu(self.linear1(input_linear.to("cuda"))))
                whole, hidden = self.encoder_rnn(input_sequence)
                reward_cur = (1 - self.compare.cos_sam(hidden, hidden_oracle).item())*10 - 1
                res.append(reward_cur)
        return res


