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

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

##NOUR
import wandb
from tqdm import tqdm
from ..client_selection.config import *
from ..clients.clientbase import Client
##RAFFOUL

class FedAvg(Server):
    def __init__(self, args, times, client_selection): ##NOUR: added CS
        super().__init__(args, times, client_selection) ##NOUR: added CS

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    ##NOUR: commented out old training
    '''
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
'''

##NOUR: changing the training for compatibility
    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            print(f"\n------------- Round {i}-------------")

            #Set up client lists
            clients_for_training = [] #for methods that require training a subset of clients before choosing some
            clients_for_aggregation = [] #final selected clients

            # Step 1: Determine the initial pool of clients for the round.
            candidate_indices = [*range(self.num_clients)]
            #clients_to_train = self.clients

            # Step 2: Candidate Selection from a larger pool (ex. for power of choice)
            # This stage selects a larger-than-needed subset for training.
            if self.args.method in CANDIDATE_SELECTION_METHOD:
                print(f"> Stage 1: Selecting {self.args.num_candidates} candidates for training (for {self.args.method}).")
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOT YET ADDED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                # For now, we assume all clients are candidates
                pass # Placeholder for candidate selection logic,, random for power of choice

            # Step 3: Pre-Selection (before requiring a training exercise)
            if self.args.method in PRE_SELECTION_METHOD:
                print(f"> Pre-training selection: choosing {self.num_clients_per_round} clients.")
                selected_indices = self.selection_method.select(
                    self.num_clients_per_round,
                    candidate_indices,
                    None # No metric needed
                )
                print("SELECTED INDICES: ",selected_indices)
                print(f"DEBUG: About to index self.clients. Length of self.clients is: {len(self.clients)}")

                # For pre-selection, the clients to be trained ARE the clients to be aggregated.
                clients_for_training = [self.clients[i] for i in selected_indices]
                clients_for_aggregation = clients_for_training # They are the same list.

            # Handle Post-Selection Methods (give a training exercise first)
            else:
                print(f"> Post-training selection enabled for method '{self.args.method}'.")
                # For post-selection, we must first train the entire pool of available clients.
                clients_for_training = self.clients

            # Step 4: Initialize methods that need the global model
            if self.args.method in NEED_INIT_METHOD:
                print(f"> Initializing selection method '{self.args.method}' with the global model.")
                # Pass the models of the clients that are about to be trained
                client_models = [c.model for c in clients_for_training]
                self.selection_method.init(self.global_model, client_models)

            # Step 4.5: Distribute the current global model to all clients who will train
            print(f"> Distributing global model to {len(clients_for_training)} clients.")
            for client in clients_for_training:
                client.set_parameters(self.global_model)
                
            # Step 5: Perform Local Training on the chosen set of clients
            # !!!!!!!!!!!!!!!CHANGE TO GIVE A SMALLER TASK SOON!!!!!!!!!
            print(f"> Sending global model to {len(clients_for_training)} clients for local training.")
            if i % self.eval_gap == 0:
                print("\nEvaluating global model before training...")
                self.evaluate(current_round=i)
                
            local_losses, accuracy, local_metrics = self.train_clients(clients_for_training)

            # Step 6: Post-Selection 
            # select clients using the results from step 5
            if self.args.method not in PRE_SELECTION_METHOD:
                print(f"> Post-training selection: choosing {self.num_clients_per_round} from {len(clients_for_training)} trained clients.")
                
                # The indices passed to select must match the list of trained clients
                trained_client_indices = [c.id for c in clients_for_training]

                # Determine which metric to pass to the selection method
                metric_for_selection = None
                if self.args.method in NEED_LOCAL_MODELS_METHOD:
                    print(f">   (Using local models as metric for {self.args.method})")
                    metric_for_selection = [c.model for c in clients_for_training]
                else:
                    print(f">   (Using local losses as metric for {self.args.method})")
                    # We need to map the collected metrics to the clients that were trained
                    metric_for_selection = local_metrics

                # The select method returns indices relative to the trained_client_indices list
                relative_selected_indices = self.selection_method.select(
                    n=self.num_clients_per_round, 
                    client_idxs=trained_client_indices,
                    metric=metric_for_selection
                )
                
                # The final set of clients for aggregation
                #self.selected_clients = [clients_for_training[i] for i in relative_selected_indices]
                clients_for_aggregation = [clients_for_training[i] for i in relative_selected_indices]

            self.selected_clients = clients_for_aggregation

            # Step 7: Aggregation
            print(f"> Aggregating models from {len(self.selected_clients)} final selected clients.")
            selected_ids = [c.id for c in self.selected_clients]
            print(f"> Selected client IDs for this round: {sorted(selected_ids)}")
            self.receive_models() # This uses self.selected_clients
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
                
