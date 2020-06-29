import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import copy

def data_mask(input):
    """
    :param input: [ , , ] one tensor
    :return: list of tensors, and it should mask one turn, list_len = len([ , , ])
    """
    dia_len = input.size(1)
    input_tensor = input.clone()
    domain_list = []
    mask_input_tensor = []
    mask_id = []
    bf_list = []

    for i in range(dia_len):
        action = input_tensor[0][i][340:549]
        domain_list.append(domain_classifier(action))
        bf = input_tensor[0][i][:340].clone()
        bf_list.append(bf)
        one = input_tensor[0][:i][:].unsqueeze(0).clone()
        mask_input_tensor.append(one)
        mask_id.append(i)

    return mask_input_tensor, mask_id, domain_list, bf_list

def domain_classifier(action):
    """
    :param action:
    :return:[ one hot encoder ]
    """
    domain = [0] * 9
    for i in range(action.shape[0]):
        if action[i].item() == 1.:
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

    return domain

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
        reward = []
        mask_input_tensor, mask_id, domain_list, input_belief = self.data_mask(input)
        r = self.forward(mask_input_tensor,torch.stack(input_belief).to("cuda"))


        return reward


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

