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

import numpy as np
import os
import torch
from torchvision import transforms
import collections
import random

"""
def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')
    
    file_path = os.path.join(data_dir, f"{idx}.pt")
    
    # Load the .pt file (returns a dictionary)
    data_dict = torch.load(file_path)

    # Extract 'samples' and convert to a PyTorch tensor
    if 'samples' not in data_dict:
        raise KeyError(f"Expected key 'samples' in the dictionary, but found {data_dict.keys()}")
    
    data = torch.tensor(data_dict['samples'], dtype=torch.float32)  # Convert numpy to tensor

    # Ensure data has the correct shape (assuming it's in (num_samples, num_channels, height, width) format)
    if data.dim() == 2:  # If it's (num_samples, features), reshape to (num_samples, 1, features)
        data = data.unsqueeze(1)  # Add a channel dimension
    elif data.dim() == 3:  # If it's (num_samples, height, width), assume grayscale and reshape
        data = data.unsqueeze(1)  # Convert to (num_samples, 1, height, width)

    # Compute mean and std **per channel only**
    data_mean = torch.mean(data, dim=(0, 2, 3), keepdim=True)  # Mean per channel
    data_std = torch.std(data, dim=(0, 2, 3), keepdim=True)  # Std per channel

    # Avoid division by zero
    data_std[data_std == 0] = 1

    # Define normalization transform
    transform = transforms.Normalize(mean=data_mean.squeeze(), std=data_std.squeeze())

    # Apply transformation
    data = transform(data)

    # Keep labels unchanged
    labels = data_dict['labels']  # Assuming labels are needed later

    #delete the last dimention
    data = data.squeeze(-1)
    
    return {"samples": data, "labels": labels}





"""

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')
        train_file = train_data_dir+'train_'+str(idx) + '.pt'
        
        # Load the .pt file using torch.load
        #train_data = torch.load(train_file,weights_only=False)  # Assuming the .pt file contains the data directly
        ##NOUR
        train_data = torch.load(train_file)
        
        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')
        test_file = test_data_dir+'test_'+str(idx) + '.pt'
        
        # Load the .pt file using torch.load
        #test_data = torch.load(test_file,weights_only=False)  # Assuming the .pt file contains the data directly
        ##NOUR
        test_data = torch.load(test_file)

        return test_data
    

##NOUR
'''
def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train=train_data['samples']
        y_train=train_data['labels']
        
        if y_train is not None and isinstance(y_train, np.ndarray):
                y_train = torch.from_numpy(y_train)
    
        # Convert to torch tensor
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze(1)
        elif len(X_train.shape) == 3 and X_train.shape[1] != 1:
            X_train = X_train.transpose(1, 2)

        
        #X_train = torch.Tensor(train_data['samples']).type(torch.float32)
        #y_train = torch.Tensor(train_data['labels']).type(torch.int64)
        """
        # Compute mean and standard deviation across (batch, sequence length)
        data_mean = torch.mean(X_train, dim=(0, 2), keepdim=True)  # Keep dimensions for broadcasting
        data_std = torch.std(X_train, dim=(0, 2), keepdim=True)

        # Avoid division by zero
        data_std = data_std + 1e-8  

        # Normalize X_train manually
        X_train = (X_train - data_mean) / data_std
        """
        
        X_train = X_train.float()
        y_train = y_train.long() if y_train is not None else None
        
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        #print('type', type(train_data))
        #print("train_data", len(train_data))
        #print("train_data", train_data[0][0].shape)
        #exit()
        
        
        #print(train_data[0][0], train_data[0][1])
        #exit()
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test=test_data['samples']
        y_test=test_data['labels']
        
        
        if y_test is not None and isinstance(y_test, np.ndarray):
                y_test = torch.from_numpy(y_test)
    
        # Convert to torch tensor
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test)
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze(1)
        elif len(X_test.shape) == 3 and X_test.shape[1] != 1:
            X_test = X_test.transpose(1, 2)
        
        #X_test = torch.Tensor(test_data['samples']).type(torch.float32)
        #y_test = torch.Tensor(test_data['labels']).type(torch.int64)
        
        
        """
        # Compute mean and standard deviation across (batch, sequence length)
        data_mean = torch.mean(X_test, dim=(0, 2), keepdim=True)  # Keep dimensions for broadcasting
        data_std = torch.std(X_test, dim=(0, 2), keepdim=True)

        # Avoid division by zero
        data_std = data_std + 1e-8  

        # Normalize X_train manually
        X_test = (X_test - data_mean) / data_std
        """
        
        X_test = X_test.float()
        y_test = y_test.long() if y_test is not None else None
        
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
'''
def read_client_data(dataset, idx, is_train=True, args=None):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = train_data['samples']
        y_train = train_data['labels']

        if args and args.imbalance_factor > 1.0:
            print(f"INFO: Simulating imbalance with factor {args.imbalance_factor} for client {idx} training data.")

            # DATA IMBAKANCE PART
            # Grouping data by class
            data_by_class = collections.defaultdict(list)
            for sample, label in zip(X_train, y_train):
                data_by_class[label].append(sample)

            # Determine target sample counts per class using exponential
            num_classes = args.num_classes
            samples_per_class = {cls: len(data_by_class[cls]) for cls in data_by_class}
            max_samples = max(samples_per_class.values()) if samples_per_class else 0

            mu = (1.0 / args.imbalance_factor) ** (1.0 / (num_classes - 1))

            imbalanced_X = []
            imbalanced_y = []

            # Subsample each class
            for class_idx in sorted(samples_per_class.keys()):
                target_count = int(max_samples * (mu**class_idx))

                # Get original data for this class
                original_samples = data_by_class[class_idx]
                original_labels = [class_idx] * len(original_samples)

                # Subsample if necessary
                if len(original_samples) > target_count:
                    indices_to_keep = random.sample(range(len(original_samples)), target_count)
                    imbalanced_X.extend([original_samples[i] for i in indices_to_keep])
                    imbalanced_y.extend([original_labels[i] for i in indices_to_keep])
                else:
                    imbalanced_X.extend(original_samples)
                    imbalanced_y.extend(original_labels)

            X_train, y_train = np.array(imbalanced_X), np.array(imbalanced_y)
            #END OF IMBALANCE PART

        if y_train is not None and isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train)

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)

        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze(1)
        elif len(X_train.shape) == 3 and X_train.shape[1] != 1:
            X_train = X_train.transpose(1, 2)

        X_train = X_train.float()
        y_train = y_train.long() if y_train is not None else None

        train_data_final = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data_final
    
    else:
        # Test data remains unchanged
        test_data = read_data(dataset, idx, is_train)
        X_test = test_data['samples']
        y_test = test_data['labels']

        if y_test is not None and isinstance(y_test, np.ndarray):
            y_test = torch.from_numpy(y_test)

        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test)

        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze(1)
        elif len(X_test.shape) == 3 and X_test.shape[1] != 1:
            X_test = X_test.transpose(1, 2)

        X_test = X_test.float()
        y_test = y_test.long() if y_test is not None else None

        test_data_final = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data_final
##RAFFOUL


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

