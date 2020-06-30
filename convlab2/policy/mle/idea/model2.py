import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import copy
from collections import defaultdict

# define the loss function, model_dialogue could also use that.
pos_weights = torch.full([549], 2, dtype=torch.float).to("cuda")
reconstruction_loss = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weights)
classification_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

def loss_fn(logp, target):
    # loss1 = reconstruction_loss(logp, target)
    loss2 = classification_loss(logp, target)
    return loss2 #, torch.mean(torch.abs(F.sigmoid(logp) - target))

def data_mask(input):
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

    for i in range(1, dia_len):
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

    def __init__(self, input_size = 549, hidden_size = 256, output_size = 9):
        super(dialogue_VAE, self).__init__()
        self.encoder_1 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.linear_belief = nn.Linear(340,hidden_size)
        self.linear_output1 = nn.Linear(hidden_size*2,hidden_size)
        self.linear_output2 = nn.Linear(hidden_size, 64)
        self.linear_output3 = nn.Linear(64, output_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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
                global_score = self.get_score_domain(input)
                reward_collc += global_score

                # data_sub.append(input)
                data_collc[last_reward].append(input)
                data_reward_collc[sum(reward_collc)].append(reward_collc)

                reward_predict += reward_collc
                # clear
                s_temp = torch.tensor([])
                a_temp = torch.tensor([])
                reward_collc = []

        reward_predict = torch.tensor(reward_predict)

        # update the reward model first.
        if needs_update:
            self.update(data_collc, 16, 32)
        return reward_predict

    def get_score_domain(self, input):
        """
        :param input: one whole dialogue
        :return: list of rewards
        """
        mask_input_tensor, mask_id, domain_list, input_belief = data_mask(input)
        r = self.forward(mask_input_tensor,torch.stack(input_belief).to("cuda"))
        r = (r>0.).float()
        reward = 0.5 - torch.sum(torch.abs(r - torch.tensor(domain_list).float().to("cuda")),1)
        reward = reward.tolist()
        reward.insert(0, 0.)
        return reward

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
            mask_input_tensor, mask_id, domain_list, input_belief = data_mask(temp)
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
                prediction = self.forward(mask_input_batch,bf_batch)
                loss, _ = loss_fn(prediction, torch.tensor(domain_batch).float().to("cuda"))
                #print("LOSS(updating for reward model): ",loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # clear stuff
                mask_input_batch = []
                mask_id_batch = []
                domain_batch = []
                bf_batch = []

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

