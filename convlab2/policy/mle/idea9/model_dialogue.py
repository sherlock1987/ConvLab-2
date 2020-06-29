import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import copy

class dialogue_VAE(nn.Module):

    def __init__(self, input_size, hidden_size = 256, output_size = 9):
        super(dialogue_VAE, self).__init__()
        self.encoder_1 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.linear_belief = nn.Linear(340,hidden_size)
        self.linear_output1 = nn.Linear(hidden_size*2,hidden_size)
        self.linear_output2 = nn.Linear(hidden_size, 64)
        self.linear_output3 = nn.Linear(64, output_size)
        self.linear1 = nn.Linear(549, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.device = "cuda"


    def forward(self, input_feature, input_belief):
        max_len = self.get_max_len(input_feature)
        original_input_tensor, padded_input_sequence, sorted_lengths, sorted_idx = self.concatenate_zero(input_feature, max_len)
        padded_input_sequence = self.linear2(F.relu(self.linear1(padded_input_sequence.to(self.device))))
        packed_input = rnn_utils.pack_padded_sequence(padded_input_sequence, sorted_lengths.data.tolist(),batch_first=True)

        _, last_h = self.encoder_1(packed_input)
        last_h = last_h.squeeze(0)
        _, reversed_idx = torch.sort(sorted_idx)
        last_h = last_h[reversed_idx]

        input_belief = self.linear_belief(input_belief)

        output = torch.cat((last_h,input_belief),dim=1)
        output = self.linear_output3(F.relu(self.linear_output2(F.relu(self.linear_output1(output)))))

        return output

    def get_score_domain(self, input):
        """
        :param input: one whole dialogue
        :return: list of rewards
        """
        len_dial = len(input[0])
        mask_input_tensor, mask_id, domain_list, input_belief = self.data_mask(input)
        output = []

        for i in range(len_dial):
            input_hidden = torch.zeros(1, self.hidden_size * self.hidden_factor).to(self.device)
            if i != 0:
                input_linear = self.linear2(F.relu(self.linear1(mask_input_tensor[i].to(self.device))))
                whole_pack, hidden = self.encoder_rnn(input_linear)
                input_hidden = whole_pack[0][mask_id[i] - 1].unsqueeze(0).to(self.device)
                input_hidden = input_hidden.view(1, self.hidden_size * self.hidden_factor)
            domain_predict = self.linear_belief(input_belief[i].unsqueeze(0).to(self.device))
            domain_predict = torch.cat((input_hidden, domain_predict), dim=1)
            domain_predict = self.linear_output2(F.relu(self.linear_output1(domain_predict)))
            target = torch.tensor(domain_list[i]).float().unsqueeze(0)
            score = 0.5 - self.bceLoss(domain_predict.cpu(), target)
            output.append(score.item())
        return output

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

    def concatenate_zero(self, input, max_len):
        """
        :param input: list [[,,] [,,] [,,] [,,] [,,] ]
        :param max_len:
        :return:pad input, pad_input_sort, sorted_length, sorted_idx
        """
        output = torch.zeros(size = (len(input), max_len, 549))

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