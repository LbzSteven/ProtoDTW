from typing import cast, Dict

import numpy as np
import torch
import sklearn.preprocessing
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tslearn.datasets import UCR_UEA_datasets
import os

class UCRDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]

        return item, label


def generate_loader(train_x, test_x, train_y, test_y, batch_size_train=64, batch_size_test=64):
    train_dataset = UCRDataset(train_x.astype(np.float32), train_y.astype(np.int64))
    test_dataset = UCRDataset(test_x.astype(np.float32), test_y.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader



def read_UCR_UEA(dataset, UCR_UEA_dataloader):
    if UCR_UEA_dataloader is None:
        UCR_UEA_dataloader = UCR_UEA_datasets()
    X_train, train_y, X_test, test_y = UCR_UEA_dataloader.load_dataset(dataset)
    # X_train = np.nan_to_num(X_train, copy=True, nan=0.0)
    # X_test = np.nan_to_num(X_test, copy=True, nan=0.0)
    if X_train is None:
        print(f"{dataset} could not load correctly")
        return None, None, None, None, None
    train_x = X_train.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    test_x = X_test.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    enc1 = sklearn.preprocessing.OneHotEncoder(sparse_output=False).fit(train_y.reshape(-1, 1))

    train_y = enc1.transform(train_y.reshape(-1, 1))
    test_y = enc1.transform(test_y.reshape(-1, 1))

    # transfer to labels starting from 0
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)
    return train_x, test_x, train_y, test_y, enc1