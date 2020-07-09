import os
import pickle
import torch
import torch.utils.data as data
from convlab2.util.multiwoz.state import default_state
from convlab2.policy.vector.dataset import ActDataset
from convlab2.util.dataloader.dataset_dataloader import MultiWOZDataloader
from convlab2.util.dataloader.module_dataloader import ActPolicyDataloader
import collections
# this is a trick of domain classify
# from domain list to one hot
encoder = {'[1. 0. 1. 0. 1. 1. 0. 0.]': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 1. 0. 0. 0. 0. 0. 0.]': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 1. 0. 1. 1. 0. 0.]': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[1. 0. 0. 0. 0. 1. 0. 0.]': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 1. 0. 0. 0. 0.]': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 1. 0. 1. 0. 0. 0. 0.]': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 1. 0. 0. 0. 0. 1.]': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 1. 0. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 0. 0. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 1. 0. 0. 0. 1. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[1. 0. 1. 0. 0. 1. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 1. 1. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 1. 0. 1. 0. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 1. 0. 0. 0. 1. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 0. 0. 0. 1.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 0. 0. 1. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 0. 1. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 1. 0. 0. 1. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[1. 0. 0. 0. 0. 0. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[1. 0. 0. 0. 1. 0. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 1. 0. 0. 0. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], '[1. 0. 0. 0. 0. 0. 1. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 0. 1. 1. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '[0. 0. 0. 1. 0. 1. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], '[0. 0. 0. 0. 1. 0. 1. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], '[1. 0. 1. 0. 0. 0. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], '[1. 0. 0. 0. 1. 1. 0. 0.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], '[1. 0. 0. 0. 0. 0. 0. 1.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], '[0. 0. 0. 0. 1. 0. 0. 1.]': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
# from one hot to domain list
decoder = {'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[1. 0. 1. 0. 1. 1. 0. 0.]', '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 1. 0. 0. 0. 0. 0. 0.]', '[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 1. 0. 1. 1. 0. 0.]', '[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[1. 0. 0. 0. 0. 1. 0. 0.]', '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 1. 0. 0. 0. 0.]', '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 1. 0. 1. 0. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 1. 0. 0. 0. 0. 1.]', '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 0. 1. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 0. 0. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 1. 0. 0. 0. 1. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[1. 0. 1. 0. 0. 1. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 0. 1. 1. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 1. 0. 1. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 1. 0. 0. 0. 1. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 0. 0. 0. 0. 1.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 0. 0. 0. 1. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 0. 0. 1. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 1. 0. 0. 1. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[1. 0. 0. 0. 0. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]': '[1. 0. 0. 0. 1. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]': '[0. 0. 1. 0. 0. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]': '[1. 0. 0. 0. 0. 0. 1. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': '[0. 0. 0. 0. 0. 1. 1. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': '[0. 0. 0. 1. 0. 1. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': '[0. 0. 0. 0. 1. 0. 1. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]': '[1. 0. 1. 0. 0. 0. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]': '[1. 0. 0. 0. 1. 1. 0. 0.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]': '[1. 0. 0. 0. 0. 0. 0. 1.]', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]': '[0. 0. 0. 0. 1. 0. 0. 1.]'}


class ActMLEPolicyDataLoader():

    def __init__(self):
        self.vector = None


    def _build_data(self, root_dir, processed_dir):
        # whole data over here.
        self.data = {}
        self.terminate = {}
        self.state_whole = {}
        self.domain = {}
        self.domain_ont_hot = {}
        action_set = set()
        data_loader = ActPolicyDataloader(dataset_dataloader=MultiWOZDataloader())
        for part in ['train', 'val', 'test']:
            self.data[part] = []
            self.terminate[part] = []
            self.state_whole[part] = []
            self.domain[part] = []
            self.domain_ont_hot[part] = []
            raw_data = data_loader.load_data(data_key=part, role='sys')[part]
            for belief_state, context_dialog_act, terminated, dialog_act in \
                    zip(raw_data['belief_state'], raw_data['context_dialog_act'], raw_data['terminated'],
                        raw_data['dialog_act']):
                # all over here.
                state = default_state()
                state['belief_state'] = belief_state
                state['user_action'] = context_dialog_act[-1]

                state['system_action'] = context_dialog_act[-2] if len(context_dialog_act) > 1 else {}

                state['terminated'] = terminated
                action = dialog_act
                # print(state['terminated'])
                self.data[part].append([self.vector.state_vectorize(state),
                                        self.vector.action_vectorize(action), state['terminated']])
                self.terminate[part].append(state['terminated'])
                sys_action = self.vector.action_vectorize(action)
                action_set.add(str(sys_action))
                self.state_whole[part].append(state)
                # current_bf = self.vector.state_vectorize(state)
                # user_action_vec = current_bf[:79]
                domain_vec = self.vector.return_stuff()
                self.domain[part].append(domain_vec)
                self.domain_ont_hot[part].append(encoder[str(domain_vec)])
        print(action_set)
        os.makedirs(processed_dir)
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.data[part], f)
        # save file of terminate 1 is not end, 0 means endding.
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}_terminate.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.terminate[part], f)
        # save state data in this file, and then use it to train, note this one is a dict file.
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}_state_whole.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.state_whole[part], f)
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}_domain_vec.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.domain[part], f)

        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}_domain_one_hot.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.domain_ont_hot[part], f)

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)

    def create_dataset(self, part, batchsz):
        print('Start creating {} dataset'.format(part))
        s = []
        a = []
        for item in self.data[part]:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = ActDataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} dataset'.format(part))
        return dataloader
