import os

import torch
from torch import optim, nn, utils
import pytorch_lightning as pl
from .transformer import Multitask_transformer
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class expressionDataset(Dataset):
    def __init__(self, x, y, isTrain, samples_weight, src_len=512, step=None):
        self.x = x
        self.y = y
        self.src_len = src_len
        self.isTrain = isTrain
        self.samples_weight = samples_weight
        self.step = step

    def __len__(self):
        if self.isTrain:
            return len(self.x) - self.src_len
            #print('hardcoded size dataloader 4')
            #return 4
        else:
            return int(np.ceil(len(self.x)/self.src_len))

    def __getitem__(self, idx):
        if self.isTrain:
            idx = idx*self.step
            output_x = np.stack(self.x[idx:(idx+self.src_len)])
            output_y = np.stack(self.y[idx:(idx+self.src_len)])
            weight = np.stack(self.samples_weight[idx:(idx+self.src_len)])
        else:
            output_x = np.stack(self.x[idx*self.src_len:(idx*self.src_len + self.src_len)])
            output_y = np.stack(self.y[idx*self.src_len:(idx*self.src_len + self.src_len)])
            weight = np.stack(self.samples_weight[idx*self.src_len:(idx*self.src_len + self.src_len)])
        return torch.from_numpy(output_x).permute(0, 3, 1, 2).float(), torch.from_numpy(
            output_y).float(), torch.from_numpy(weight).float()


def getDataloader(x, y, isTrain, batch_size, window_length, samples_weight, step=None):
    if isTrain:
        dataset = expressionDataset(x, y, True, samples_weight, window_length, step)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = expressionDataset(x, y, False, samples_weight, window_length, step)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader



