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

#!/usr/bin/env python

'''NOTE from Nour Raffoul: I start any block of code I added by ##NOUR and end it with ##RAFFOUL'''

import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbn import FedBN
from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
import random

##NOUR
import sys
import time
from utils.argparse import get_args
from flcore.client_selection import *

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
project_root = os.path.dirname(current_dir)
# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning to prioritize

AVAILABLE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    AVAILABLE_WANDB = False
##RAFFOUL

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#fix_randomness(0)


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

#DEFINING A RUN
def run(args,client_selection):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        ##NOUR
        #GENERATE THE MODEL, I wrapped in a function for clarity
        args.model = generate_model(model_str,args)
        print(args.model)

        #INITIALIZE CS
        print(f"Initializing client selection method: {args.method}")
        args.selection_method = client_selection_method(args) # Use helper
        ##RAFFOUL

        # select algorithm + DEFINE SERVER
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i, client_selection)
        if args.algorithm == "FedBn":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBN(args, i, client_selection)
        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()

##NOUR
def client_selection_method(args):
    #total = args.total_num_client if args.num_available is None else args.num_available
        kwargs = {'total': args.total_num_clients, 'device': args.device}
        if args.method == 'Random':
            return RandomSelection(**kwargs)
        elif args.method == 'AFL':
            return ActiveFederatedLearning(**kwargs, args=args)
        elif args.method == 'Cluster1':
            return ClusteredSampling1(**kwargs, n_cluster=args.num_clients_per_round)
        elif args.method == 'Cluster2':
            return ClusteredSampling2(**kwargs, dist=args.distance_type)
        elif args.method == 'Pow-d':
            assert args.num_candidates is not None
            return PowerOfChoice(**kwargs, d=args.num_candidates)
        elif args.method == 'DivFL':
            assert args.subset_ratio is not None
            return DivFL(**kwargs, subset_ratio=args.subset_ratio)
        elif args.method == 'GradNorm':
            return GradNorm(**kwargs)
        ##NOUR: implementing full participation as a benchmark
        elif args.method == 'FullPart':
            args.num_clients_per_round=args.total_num_clients
            return FullPart(**kwargs)
        elif args.method == 'RandomSelect':
            return RandomSelect(**kwargs)
        else:
            raise('CHECK THE NAME OF YOUR SELECTION METHOD')
        
def generate_model(model_str,args):
    if model_str == "MLR": # convex
            if "MNIST" in args.dataset:
                return Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                return Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                return Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

    elif model_str == "CNN": # non-convex
            if "MNIST" in args.dataset:
                return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                return Digit5CNN().to(args.device)
            elif "EEG" in args.dataset:
                configs=EEG()
                return EEGCNN(configs).to(args.device)
            else:
                return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

    elif model_str == "DNN": # non-convex
            if "MNIST" in args.dataset:
                return DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                return DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
               return DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
    elif model_str == "ResNet18":
            return torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        
    elif model_str == "ResNet10":
            return resnet10(num_classes=args.num_classes).to(args.device)
        
    elif model_str == "ResNet34":
            return torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "AlexNet":
           return alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
    elif model_str == "GoogleNet":
            return torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "MobileNet":
            return mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
    elif model_str == "LSTM":
            return LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

    elif model_str == "BiLSTM":
            return BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim).to(args.device)

    elif model_str == "fastText":
            return fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

    elif model_str == "TextCNN":
            return TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)

    elif model_str == "Transformer":
            return TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
    elif model_str == "AmazonMLP":
            return AmazonMLP().to(args.device)

    elif model_str == "HARCNN" :
            if args.dataset == 'HAR' :
                return HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            if args.dataset == "HHAR":
                return HARCNN(128, dim_hidden=64, num_classes=args.num_classes, conv_kernel_size=(1, 1), 
                                    pool_kernel_size=(1, 1)).to(args.device)
            if args.dataset == "FD":
                return HARCNN(5120, dim_hidden=64, num_classes=args.num_classes, conv_kernel_size=(1, 1), 
                                    pool_kernel_size=(1, 1)).to(args.device)
            if args.dataset == "WISDM":
                return HARCNN(128, dim_hidden=64, num_classes=args.num_classes, conv_kernel_size=(1, 1), 
                                    pool_kernel_size=(1, 1)).to(args.device)
            if args.dataset == "ToyDataset":
                return HARCNN(100, dim_hidden=64, num_classes=args.num_classes, conv_kernel_size=(1, 1), 
                                    pool_kernel_size=(1, 1)).to(args.device)
            if args.dataset == "EEG":
                return EEGCNN(EEG()).to(args.device)
            elif args.dataset == 'PAMAP2':
                return HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)
    else:
            raise NotImplementedError
##RAFFOUL

#MAIN FUNCTION
if __name__ == "__main__":
    total_start = time.time()

    ##NOUR: GET ARGUMENTS FROM PARSER
    args=get_args()

    ##ADD SOME REDUNDANT ARGUMENTS FOR COMPATIBILITY WITH CS METHODS
    args.total_num_clients=args.num_clients
    args.num_clients_per_round=int(args.join_ratio*args.num_clients)
    args.num_epoch = args.local_epochs

    # save to wandb
    args.wandb = AVAILABLE_WANDB
    if args.wandb:
        wandb.init(
            project=f'AFL-{args.dataset}-{args.num_clients_per_round}-{args.num_available}-{args.total_num_clients}',
            name=f"{args.method}{args.comment}",
            config=args,
            dir='.',
            save_code=True
        )
        wandb.run.log_code(".", include_fn=lambda x: 'src/' in x or 'main.py' in x)
    ##RAFFOUL

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:

    client_selection = client_selection_method(args)
    run(args, client_selection)
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
