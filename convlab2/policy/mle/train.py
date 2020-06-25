import os
import torch
import logging
import numpy as np
from convlab2.util.train_util import to_device
import torch.nn as nn
from torch import optim
from convlab2.policy.mle.idea3.idea_3_max_margin import Reward_max_margin
import matplotlib.pyplot  as plt
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from convlab2.policy.mle.idea4.autoencoder import auto_encoder

class Reward_predict(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Reward_predict, self).__init__()
        self.encoder_1 = nn.LSTM(input_size, output_size, batch_first=True, bidirectional=False)
        self.encoder_2 = nn.LSTM(output_size, output_size)

        self.m = nn.Sigmoid()
        self.loss = nn.BCELoss(size_average=False, reduce=True)
        self.cnn_belief = nn.Linear(input_size - output_size, output_size)
        self.cnn_output = nn.Linear(output_size, output_size)

    def forward(self, input_feature, input_belief, target):
        # to construct the batch first, then we could compute the loss function for this stuff, simple and easy.
        _, (last_hidden, last_cell) = self.encoder_1(input_feature)
        # second Part

        _, (predict_action, last_cell) = self.encoder_2(self.cnn_belief(input_belief), (last_hidden, last_cell))

        loss = self.loss(self.m(self.cnn_output(predict_action)), target)
        return loss


class MLE_Trainer_Abstract():
    def __init__(self, manager, cfg):
        self._init_data(manager, cfg)
        self.policy = None
        self.policy_optim = None
        # this is for fake data generator
        self.generator_fake = None
        # define the stuff from the reward machine

    def _init_data(self, manager, cfg):
        self.data_train = manager.create_dataset('train', cfg['batchsz'])
        self.data_valid = manager.create_dataset('val', cfg['batchsz'])
        self.data_test = manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        self.loss_record = []
        # stuff for idea 2
        self.reward_predictor = Reward_predict(549, 457, 209)
        self.reward_optim = optim.Adam(self.reward_predictor.parameters(), lr=1e-4)

        # stuff for idea 3
        self.reward_predictor_idea3 = Reward_max_margin(549, 209)
        self.reward_optim_idea3 = optim.Adam(self.reward_predictor_idea3.parameters(), lr=1e-4)
        #    init the terminate state and use if when training our model.
        self.terminate_train = {}
        self.state_whole = {}
        self.success = []
        self.success_plot = []
        # load data of terminate
        for part in ['train', 'val', 'test']:
            with open(os.path.join("//home//raliegh//图片//ConvLab-2//convlab2//policy//mle//multiwoz//processed_data",
                                   '{}_terminate.pkl'.format(part)), 'rb') as f:
                self.terminate_train[part] = pickle.load(f)
        # load data of state_whole
        for part in ['train', 'val', 'test']:
            with open(os.path.join("//home//raliegh//图片//ConvLab-2//convlab2//policy//mle//multiwoz//processed_data",
                                   '{}_state_whole.pkl'.format(part)), 'rb') as f:
                self.state_whole[part] = pickle.load(f)

    def policy_loop(self, data):
        # this is from states to predict the a, pretty similar to idea2
        s, target_a = to_device(data)
        a_weights = self.policy(s)

        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a

    def reward_training(self, epoch):
        self.reward_predictor.train()
        s_temp = torch.tensor([])
        a_temp = torch.tensor([])
        loss = torch.tensor([0]).float()

        for i, data in enumerate(self.data_train):
            # curr_state is everything, contains domain, action, bf, and also user action.
            curr_state = self.state_whole["train"][i]
            fake_action = self.generator_fake.predict(curr_state)
            s, a = to_device(data)
            # s_temp = s[:i+1]
            try:
                s_temp = torch.cat((s_temp, s), 0)
                a_temp = torch.cat((a_temp, a), 0)
            except Exception as e:
                s_temp = s
                a_temp = a

            s_train = s_temp.unsqueeze(0)
            a_train = a_temp.unsqueeze(0)
            # print(s_train.shape)
            s_train_np = np.array(s_train)
            if len(s_train[0]) >= 2:
                # print("-"*300)
                input_pre = torch.cat((s_train, a_train), 2)[0][:-1].unsqueeze(0)
                input_bf = s_train[0][-1].unsqueeze(0).unsqueeze(0)
                target = a_train[0][-1].unsqueeze(0).unsqueeze(0)
                # print(input_pre.shape,input_bf.shape,target.shape)

                terminate = self.terminate_train["train"][i]
                if terminate == False:
                    micro_loss = self.reward_predictor(input_pre, input_bf, target)
                    loss += micro_loss
                else:
                    # predict the last one and then loss backward and then clear the button
                    micro_loss = self.reward_predictor(input_pre, input_bf, target)
                    loss += micro_loss
                    # print(loss,loss/(len(s_temp)-3))
                    if loss != torch.tensor([0]).float():
                        # print(loss.item()/(len(iteration)-3),loss.shape)
                        loss.backward()
                        for name, param in self.reward_predictor.named_parameters():
                            if "cnn" not in name:
                                # print(name)
                                # print(param.grad)
                                pass
                        self.reward_optim.step()
                        self.reward_optim.zero_grad()
                        self.loss_record.append(loss.item())
                        # clear the button
                        s_temp = torch.tensor([])
                        a_temp = torch.tensor([])
                        loss = torch.tensor([0]).float()
        #         remember to save the model

        if (epoch + 1) % self.save_per_epoch == 0:
            self.reward_model_save(self.save_dir, epoch)
            axis = [i for i in range(len(self.loss_record))]
            plt.plot(axis, self.loss_record)
            plt.xlabel('Number of turns')
            plt.ylabel('Embedding Loss')
            plt.show()

    def reward_training_idea_3(self, epoch):
        self.reward_predictor_idea3.train()
        s_temp = torch.tensor([])
        a_temp = torch.tensor([])
        loss = torch.tensor([0]).float()


        for i, data in enumerate(self.data_train):
            # curr_state is everything, contains domain, action, bf, and also user action.
            curr_state = self.state_whole["train"][i]
            fake_action = self.generator_fake.predict(curr_state)
            s, a = to_device(data)
            # s_temp = s[:i+1]
            try:
                s_temp = torch.cat((s_temp, s), 0)
                a_temp = torch.cat((a_temp, a), 0)
            except Exception as e:
                s_temp = s
                a_temp = a
            # [ , , ]
            s_train = s_temp.unsqueeze(0)
            a_train = a_temp.unsqueeze(0)
            # print(s_train.shape)
            if len(s_train[0]) >= 2:
                # print("-"*300)
                input_real = torch.cat((s_train, a_train), 2)[0][:-1].unsqueeze(0)
                # construct the data from fake
                #
                a_train_pre = a_train[0][:-1]
                fake_a = fake_action.unsqueeze(0).float()
                a_train_fake = torch.cat((a_train_pre,fake_a),0)
                input_fake = torch.cat((s_train,a_train_fake.unsqueeze(0)),2)
                # constrcut stuff for advantage LSTM
                s_train_pre = s_train[0][:-1]
                # [ , , ]
                input_pre = torch.cat((s_train_pre,a_train_pre),1).unsqueeze(0)
                s_last = s_train[0][-1]
                a_last = a_train[0][-1]
                input_last_real = torch.cat((s_last,a_last)).unsqueeze(0).unsqueeze(0)
                input_last_fake = torch.cat((s_last.unsqueeze(0),fake_a),1).unsqueeze(0)
                # print(input_pre.shape,input_bf.shape,target.shape)
                terminate = self.terminate_train["train"][i]

                if terminate == False:
                    """
                    micro_loss, res_1, res_2 = self.reward_predictor_idea3.loss(input_real,input_fake,a_temp[-1].unsqueeze(0),fake_a)
                    self.success.append(res_1)
                    self.success.append(res_2)
                    if len(self.success) == 100:
                        curr_res = np.sum(self.success)/100
                        print("fail: ", curr_res)
                        self.success_plot.append(curr_res)
                        self.success = []
                    """
                    # """
                    # method 2
                    micro_loss, res = self.reward_predictor_idea3.loss_plus_lstm(input_real,input_fake)
                    self.success.append(res)
                    if len(self.success) == 100:
                        curr_res = np.sum(self.success)/100
                        print("fail: ", curr_res)
                        self.success_plot.append(curr_res)
                        self.success = []
                    # """
                    loss += micro_loss

                else:
                    # predict the last one and then loss backward and then clear the button
                    """
                    micro_loss, res_1, res_2 = self.reward_predictor_idea3.loss(input_real,input_fake,a_temp[-1].unsqueeze(0),fake_a)
                    self.success.append(res_1)
                    self.success.append(res_2)
                    if len(self.success) == 100:
                        curr_res = np.sum(self.success)/100
                        print("fail: ", curr_res)
                        self.success_plot.append(curr_res)
                        self.success = []
                    """
                    # """
                    # method 2
                    micro_loss, res = self.reward_predictor_idea3.loss_plus_lstm(input_real,input_fake)
                    self.success.append(res)
                    if len(self.success) == 100:
                        curr_res = np.sum(self.success)/100
                        print("fail: ", curr_res)
                        self.success_plot.append(curr_res)
                        self.success = []
                    # """
                    loss += micro_loss
                    len_dia = len(s_temp)
                    # print(loss.item()/len_dia)
                    if loss != torch.tensor([0]).float():
                        # print(loss, loss.dtype)
                        loss.backward()
                        # to check if still have gradients
                        # for name, param in self.reward_predictor_idea3.named_parameters():
                        #     if "cnn" not in name:
                        #         print(name)
                        #         print(param.grad)
                        self.reward_optim_idea3.step()
                        self.reward_optim_idea3.zero_grad()
                        self.loss_record.append(loss.item()/len_dia)
                        # clear the button
                        s_temp = torch.tensor([])
                        a_temp = torch.tensor([])
                        loss = torch.tensor([0]).float()
        #         remember to save the model

        if (epoch + 1) % self.save_per_epoch == 0:
            self.reward_model_save_idea3(self.save_dir, epoch)
            print("total fail rate",np.sum(self.success)/len(self.success))
            print(self.success)
            print(self.success_plot)
            plot_stuff = self.success_plot
            # plot
            axis = [i for i in range(len(plot_stuff))]
            plt.plot(axis, plot_stuff)
            plt.xlabel('Number of dialogues')
            plt.ylabel('Embedding Loss')
            plt.show()

    def auto_encoder_training(self,epoch):
        s_temp = torch.tensor([])
        a_temp = torch.tensor([])
        data_list = []
        data_tensor = torch.tensor([])
        for i, data in enumerate(self.data_train):
            s, a = to_device(data)
            try:
                s_temp = torch.cat((s_temp, s), 0)
                a_temp = torch.cat((a_temp, a), 0)
            except Exception as e:
                s_temp = s
                a_temp = a
            # [ , , ]
            s_train = s_temp.unsqueeze(0)
            a_train = a_temp.unsqueeze(0)
            # print(s_train.shape)
            if len(s_train[0]) >= 2:
                input_real = torch.cat((s_train, a_train), 2)
                terminate = self.terminate_train["train"][i]
                if terminate == False:
                    """
                    For simlicity, here I will not implement the auto encoder only for last stage.
                    """
                    pass
                else:
                    # predict the last one and then loss backward and then clear the button
                    """
                    predict, compute loss, and went forward.
                    """
                    # """
                    data_list.append(input_real)
                    pass
                    # """
                        # clear the button
                    s_temp = torch.tensor([])
                    a_temp = torch.tensor([])
                    input_real = torch.tensor([])

        print("finish creating dataset for auto-encoder")
        print("start training auto-encoder")
        auto_encoder(data_list)
        if (epoch + 1) % self.save_per_epoch == 0:
            self.reward_model_save_idea3(self.save_dir, epoch)
            print("total fail rate", np.sum(self.success) / len(self.success))
            print(self.success)
            print(self.success_plot)
            plot_stuff = self.success_plot
            # plot
            axis = [i for i in range(len(plot_stuff))]
            plt.plot(axis, plot_stuff)
            plt.xlabel('Number of dialogues')
            plt.ylabel('Embedding Loss')
            plt.show()

    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            terminate = self.terminate_train["train"][i]
            self.policy_optim.zero_grad()
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            self.policy_optim.step()

            if (i + 1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.

        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.eval()

    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """
        a_loss = 0.
        for i, data in enumerate(self.data_valid):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_valid)
        logging.debug('<<dialog policy>> validation, epoch {}, loss_a:{}'.format(epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = a_loss
            self.save(self.save_dir, 'best')

        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy>> test, epoch {}, loss_a:{}'.format(epoch, a_loss))
        return best

    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN

        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_test):
            s, target_a = to_device(data)
            a_weights = self.policy(s)
            a = a_weights.ge(0)
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN

        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        print(a_TP, a_FP, a_FN, F1)

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_mle.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def reward_model_save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.reward_predictor.state_dict(), directory + '/' + str(epoch) + 'reward_mle.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def reward_model_save_idea3(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.reward_predictor_idea3.state_dict(), directory + '/' + str(epoch) + 'reward3_mle.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))