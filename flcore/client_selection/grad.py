from .client_selection import ClientSelection
import torch
import numpy as np
from itertools import product
import sys


'''GradNorm Client Selection'''
class GradNorm(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

    def select(self, n, client_idxs, metric, round=0, results=None):
        local_models = metric
        confs = []
        for local_model in local_models:
            local_grad = local_model.linear_2.weight.grad.data #head.conv.weight.grad.data
            local_grad_norm = torch.sum(torch.abs(local_grad)).cpu().numpy()
            confs.append(local_grad_norm)

        ood_scores = np.array(confs).reshape(-1)
        # high uncertainty (high ood score)
        selected_client_idxs = np.argsort(ood_scores)[-n:]
        return selected_client_idxs.astype(int)

##NOUR: following is full participation and random participation as benchmarks
##NOUR: CODE STARTS HERE
'''Full Participation'''
class FullPart(ClientSelection): 
    def __init__(self, total, device):
        super().__init__(total, device)

    def select(self, n, client_idxs, metric, round=0, results=None):
        return client_idxs.astype(int) if isinstance(client_idxs, np.ndarray) else np.array(client_idxs).astype(int)

'''Random Selection'''       
class RandomSelect(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

    def select(self, n, client_idxs, metric=None, round=0, results=None):
        # Ensure client_idxs is a NumPy array for easy random selection
        #available_client_idxs = np.array(client_idxs)

        # Randomly select 'n' clients from the available ones
        # Assuming n <= len(available_client_idxs) for valid random selection.
        selected_client_idxs = np.random.choice(client_idxs, n, replace=False)
        return selected_client_idxs.astype(int)

##NOUR: CODE ENDS HERE
def progressBar(idx, total, bar_length=20):

    percent = float(idx) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r> compute similarity: [{}] {}% ({}/{})".format(arrow + spaces, int(round(percent * 100)),
                                                                       idx, total))
    sys.stdout.flush()
