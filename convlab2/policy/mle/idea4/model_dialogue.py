import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

class dialogue_VAE(nn.Module):
    def __init__(self, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

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
        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.output_layer = nn.Linear(hidden_size * (2 if bidirectional else 1), embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

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
        original_input_tensor, pack_input_sequence, sorted_lengths, sorted_idx = self.concatenate_zero(input_sequence, max_len)
        #pack_input_sequence = self.relu(self.linear1(pack_input_sequence.to("cuda")))
        packed_input = rnn_utils.pack_padded_sequence(pack_input_sequence.to("cuda"), sorted_lengths.data.tolist(), batch_first=True)

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

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        # why dropout this one.
        # In fact, it is going to make this stuff work.

        # decoder forward pass
        # hidden comes from encoder, coming from laten variable.
        # hidden [ , , ] three dimention

        input_dropout = self.dropout(original_input_tensor)#self.relu(self.linear1(original_input_tensor.to("cuda"))))
        # pack stuff
        packed_input = rnn_utils.pack_padded_sequence(input_dropout.to("cuda"), sorted_lengths.data.tolist(), batch_first=True)

        outputs, _ = self.decoder_rnn(packed_input, hidden)
        # output layer first!
        # unpack stuff

        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True,padding_value=0.0)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        # b,s,_ = padded_outputs.size()

        outputs_layer = self.sigmoid(self.output_layer(padded_outputs))

        # z is the distribution.
        # outputs = self.output_layer(padded_outputs)
        return original_input_tensor, outputs_layer, mean, logv, z

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

