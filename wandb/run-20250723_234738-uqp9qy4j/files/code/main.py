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

'''NOTE from Nour Raffoul: I start any block of code I added by ##NOUR and end it with ##RAFFOUL'''

#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging
from torch.utils.data import DataLoader
from flcore.servers.serveravg import FedAvg
import torch.nn.functional as F
from utils.data_utils import read_client_data
from flcore.algorithms.algorithms import get_algorithm_class
from  configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
import random
from flcore.algorithms.models import CNN,classifier


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
    #np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#fix_randomness(1)


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

#DEFINING A RUN
#NOUR: added client selection as an argument
def run(args,client_selection):

    time_list = []
    reporter = MemReporter()
    #DECIDING SOURCE CLIENT
    #args.src_id=np.random.randint(0,args.num_clients)
    #args.src_id=8
    print(f"\n****** Client source: {args.src_id} *******")
    batch_size=args.batch_size
    train_data = DataLoader(read_client_data(args.dataset, args.src_id, is_train=True), batch_size, drop_last=True, shuffle=True) 
    test_data = DataLoader(read_client_data(args.dataset, args.src_id, is_train=False), batch_size, drop_last=True, shuffle=True) 
    
    dataset_class = get_dataset_class(args.dataset)()
    hparams_class = get_hparams_class(args.dataset)()
    hparams = {**hparams_class.alg_hparams,**hparams_class.train_params}
   
    algorithm_class = get_algorithm_class("SHOT")
    args.shot = algorithm_class(CNN, dataset_class, hparams, args.device)
    args.shot.to(args.device)
    args.shot.pretrain(src_dataloader=train_data)
    test_acc=0

    with torch.no_grad():
        for data, y in test_data:
            data = data.float().to(args.device)
            y = y.view((-1)).long().to(args.device)

            # forward pass
            features, seq_features = args.shot.feature_extractor(data)
            output = args.shot.classifier(features)

            # compute loss
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            total = len(test_data.dataset)
            p=test_acc / total * 100
        print(f'Accuracy of source client: {p:.2f}%')

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        ##NOUR
        #GENERATE THE MODEL, I wrapped in a function for clarity
        #args.model = generate_model(model_str,args)
        print(args.model)

        #INITIALIZE CS
        print(f"Initializing client selection method: {args.method}")
        args.selection_method = client_selection_method(args) # Use helper
        
        ##RAFFOUL
        # select algorithm
        if args.algorithm == "FedAvg":
            args.model = args.shot
            #server = FedAvg(args, i)
            ##NOUR: added CS as an arg
            server = FedAvg(args, i, client_selection)
            
        # if args.algorithm == "FedBn":
        #     args.head = copy.deepcopy(args.model.fc)
        #     args.model.fc = nn.Identity()
        #     args.model = BaseHeadSplit(args.model, args.head)
        #     server = FedBN(args, i)
        server.train()

        # time_list.append(time.time()-start)

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
##RAFFOUL

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

    client_selection = client_selection_method(args) ##NOUR
    run(args, client_selection)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
