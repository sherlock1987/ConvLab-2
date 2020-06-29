import torch
import torch.nn as nn
import torch.nn.functional as F

class Reward_predict(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Reward_predict, self).__init__()
        self.encoder_1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.linear_belief = nn.Linear(input_size-output_size,hidden_size)
        self.linear_output1 = nn.Linear(hidden_size*2,output_size)
        self.linear_output2 = nn.Linear(output_size, output_size)

        self.m = nn.Sigmoid()
        self.loss = nn.BCELoss(size_average= False,reduce= True)

    def forward(self, input_feature, input_belief, target):
        _, (_, last_cell) = self.encoder_1(input_feature)
        input_belief = self.linear_belief(input_belief)
        output = torch.cat((last_cell,input_belief),dim=0)
        output = F.relu(self.linear_output1(output))

        loss = self.loss(self.m(self.linear_output2(output)),target)
        return loss

    def compute_reward(self, input_feature, input_belief, input_predict_RL):
        # compute the reward based on the very easy methocd: product of two vectors
        _, (last_hidden, last_cell) = self.encoder_1(input_feature)
        # second Part
        _, (predict_action, last_cell) = self.encoder_2(self.cnn_belief(input_belief), (last_hidden, last_cell))
        action_prob = self.m(self.cnn_output(predict_action))
        reward = action_prob.unsqueeze(0) * input_predict_RL.unsqueeze(0)
        res = torch.sum(reward.unsqueeze(0).unsqueeze(0))
        return res

    # def bellman_equation(self,r, mask, gamma):
    #     """
    #     we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
    #     :param r: reward, Tensor, [b]
    #     :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
    #     :return: V-target(s), Tensor
    #     """
    #     batchsz = r.size(0)
    #
    #     # v_target is worked out by Bellman equation.
    #     v_target = torch.Tensor(batchsz)
    #
    #     prev_v_target = 0
    #     for t in reversed(range(batchsz)):
    #         # mask here indicates a end of trajectory
    #         # this value will be treated as the target value of value network.
    #         # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
    #         # formula: V(s_t) = r_t + gamma * V(s_t+1)
    #         v_target[t] = r[t] + gamma * prev_v_target * mask[t]
    #         # update previous
    #         prev_v_target = v_target[t]
    #
    #     return v_target