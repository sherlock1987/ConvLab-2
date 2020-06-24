import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken = 549, ninp = 256, nhead = 2, nhid = 200, nlayers = 2, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.input_linear = nn.Linear(ntoken, ninp)
        self.ninp = ninp

        self.decoder_rnn = nn.GRU(ninp, nhid, batch_first=True)
        self.domain_classify_layer1 = nn.Linear(ninp, 64)
        self.domain_classify_layer2 = nn.Linear(64,16)
        self.domain_classify_layer3 = nn.Linear(16,7)

        self.output_linear = nn.Linear(nhid, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        batch_size = len(src)
        # ENCODER
        original_src, padded_src, sorted_lengths, sorted_idx = self.concatenate_zero(src, max_len)
        padded_src_for_decoder = padded_src.clone()

        src = original_src.to("cuda")
        src = self.input_linear(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        hidden = output[0] # batch * feature size
        disc_res = self.domain_classify_layer3(F.relu(self.domain_classify_layer2(F.relu(self.domain_classify_layer1(hidden)))))

        hidden = hidden.unsqueeze(0)

        padded_src_for_decoder = self.linear2(F.relu(self.linear1(padded_src_for_decoder.to("cuda"))))
        padded_src_for_decoder = self.dropout(padded_src_for_decoder)

        # pack stuff
        packed_input = rnn_utils.pack_padded_sequence(padded_src_for_decoder, sorted_lengths.data.tolist(), batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden)
        # unpack stuff
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True,padding_value=0.0)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        output = self.output_linear(padded_outputs)

        return output

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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
