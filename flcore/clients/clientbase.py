# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import random




class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        
        #self.model = args.model ##NOUR: replace with next line
        self.model = copy.deepcopy(args.model)

        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        
        self.num_classes = args.num_classes

        ##NOUR
        ##train_samples now contains data not lengths
        #self.train_samples = train_samples
        #self.test_samples = test_samples
        self.train_samples = len(train_samples)
        self.test_samples = len(test_samples)
        ##RAFFOUL

        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        ##NOUR
        ##NEW ARGS
        self.args = args ##for imbalance
        self.client_idx = self.id
        self.test_data = self.test_samples
        #self.trainer = Trainer(model, args)
        self.num_epoch = args.num_epoch  # E: number of local epoch
        #self.nTrain = nTrain
        self.loss_div_sqrt = args.loss_div_sqrt
        self.loss_sum = args.loss_sum

        ##FOR STORING PRE-LOADED DATA
        #train_samples is now being passed as the actual data, not its length
        self.local_train_data = train_samples
        self.local_test_data = test_samples
        self.train_samples = len(train_samples)
        self.test_samples = len(test_samples)
        ##RAFFOUL

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.feature_extractor.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.feature_extractor.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self.learning_rate_scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.learning_rate_decay_gamma)
        self.optimizer = torch.optim.SGD(self.model.feature_extractor.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
             optimizer=self.optimizer, 
             gamma=args.learning_rate_decay_gamma
         )
        
        
        
        # optimizer
        
        self.learning_rate_decay = args.learning_rate_decay
        
    def get_client_idx(self):
        return self.client_idx
    
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        ##NOUR: nvm, preloading data
        '''
        ##NOUR: pass args to read_client_data
        #train_data = read_client_data(self.dataset, self.id, is_train=True)
        train_data = read_client_data(self.dataset, self.id, is_train=True, args=self.args)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
        '''
        return DataLoader(self.local_train_data, batch_size, drop_last=True, shuffle=True)
        ##RAFFOUL

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        ##NOUR: using preloaded data
        '''
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        '''
        return DataLoader(self.local_test_data, batch_size, drop_last=False, shuffle=True)
    
    def set_parameters(self, model):
        for new_param, old_param in zip(model.feature_extractor.parameters(), self.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.feature_extractor.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.feature_extractor.parameters(), new_params):
            param.data = new_param.data.clone()



    ##NOUR: testing on global model instead of local to fix issues
    #def test_metrics(self):  ##NOUR: adding a model variable to the function signature to pass the global model
    def test_metrics(self, test_model, current_round=0):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)

        #self.model.eval()
        ##NOUR: use global model
        test_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        #y_true = []
        ##NOUR: extra step in calculating the arrays
        y_prob_list = []
        y_true_list = []
        ##RAFFOUL
        total_test_loss = 0.0 #NOUR: added this for loss to check if it goes up again
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                ##NOUR: using global model again
                #features, seq_features = self.model.feature_extractor(x)
                #output = self.model.classifier(features)
                features, seq_features = test_model.feature_extractor(x)
                output = test_model.classifier(features)

                #adding loss update
                loss = self.loss(output, y)
                total_test_loss += loss.item() * y.shape[0]
                ##RAFFOUL

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                
                ##NOUR
                '''y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)'''
                y_prob_list.append(output.detach().cpu().numpy())
                y_true_list.append(y.detach().cpu().numpy())
                ##RAFFOUL

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        ##NOUR
        #y_prob = np.concatenate(y_prob, axis=0)
        #y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob_list, axis=0)
        y_true_raw = np.concatenate(y_true_list, axis=0)
        #print(f"Client {self.id}, Final raw shapes: y_prob={y_prob.shape}, y_true_raw={y_true_raw.shape}") ##DEBUGGING
        #Binarize here instead
        y_true_binarized = label_binarize(y_true_raw, classes=range(self.num_classes))
        # This handles a client's test set missing some classes
        if y_true_binarized.shape[1] != self.num_classes:
             y_true_full = np.zeros((y_true_binarized.shape[0], self.num_classes))
             for i, label in enumerate(y_true_raw):
                 y_true_full[i, label] = 1
             y_true_binarized = y_true_full
        ##RAFFOUL


        #print the shapes
        #print('y_prob shape:', y_prob.shape)  
        #print('y_true shape:', y_true.shape)        
  
        ##NOUR: changing the variable
        #auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auc = metrics.roc_auc_score(y_true_binarized, y_prob, average='micro')
        
        #return test_acc, test_num, auc
        return test_acc, test_num, auc, total_test_loss #NOUR: added loss to returned metrics


    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
          
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                features, seq_features = self.model.feature_extractor(x)
                output = self.model.classifier(features)
                # output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
               
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
