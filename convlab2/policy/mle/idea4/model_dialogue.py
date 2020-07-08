import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from convlab2.policy.mle.idea4.utils import to_var
import copy
class dialogue_VAE(nn.Module):
    def __init__(self, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                 num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

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
    # computation func.
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
        :param input_sequence: list of tensors
        :param length:
        :return:
        """
        # order
        batch_size = len(input_sequence)
        max_len = self.get_max_len(input_sequence)

        # ENCODER
        # pack stuff and unpack stuff later.
        original_input_tensor, padded_input_sequence, sorted_lengths, sorted_idx = self.pad(input_sequence, max_len)
        # padded_input_sequence = self.linear2(self.relu(self.linear1(padded_input_sequence.to("cuda"))))
        padded_input_sequence_decoder = padded_input_sequence.clone()
        padded_input_sequence = self.linear2(self.relu(self.linear1(padded_input_sequence.to("cuda"))))

        packed_input = rnn_utils.pack_padded_sequence(padded_input_sequence.to("cuda"), sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        # related to latent size, which is 16 (16/256)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        # DECODER latent to real hidden states
        hidden = self.latent2hidden(z)

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

        #input_dropout = self.dropout(original_input_tensor)
        input_dropout = self.linear2(self.relu(self.linear1(padded_input_sequence_decoder.to("cuda"))))

        # input_dropout = self.dropout(self.linear2(self.relu(self.linear1(original_input_tensor.clone().to("cuda")))))
        # pack stuff
        input_dropout = self.dropout(input_dropout)
        packed_input = rnn_utils.pack_padded_sequence(input_dropout.to("cuda"), sorted_lengths.data.tolist(), batch_first=True)

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
        return original_input_tensor, outputs_layer, mean, logv, z

    def pad(self, input, max_len):
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

    def compress(self, input):

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

        return hidden

    def compress_eval(self, input):
        """
        :param input: network tends to converge, but I am still confused which part should I use. Haha.
        :return:
        """
        input = self.linear2(self.relu(self.linear1(input.to("cuda"))))

        _, hidden = self.encoder_rnn(input)

        # if self.bidirectional or self.num_layers > 1:
        #     # flatten hidden state
        #     hidden = hidden.view(1, self.hidden_size * self.hidden_factor)
        # else:
        #     hidden = hidden.squeeze()
        #
        # # REPARAMETERIZATION
        # # related to latent size, which is 16 (16/256)
        # mean = self.hidden2mean(hidden)
        # logv = self.hidden2logv(hidden)
        # std = torch.exp(0.5 * logv)
        #
        # z = to_var(torch.randn([1, self.latent_size]))
        # distribution = z * std + mean

        return _, hidden