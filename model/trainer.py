import random

import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader


class expressionDataset(Dataset):
    def __init__(self, x, y, isTrain, samples_weight, src_len=512, step=None, expression_len=0):
        self.x = x
        self.y = y
        self.src_len = src_len
        self.isTrain = isTrain
        self.samples_weight = samples_weight
        self.step = step
        self.expression_len = expression_len
        print('change dataset: only use iou=1 window!')

    def __len__(self):
        if self.isTrain:
            return int((len(self.x) - self.src_len) / self.step)
            #print('hardcoded size dataloader 4')
            #return 4
        else:
            #return int(np.ceil(len(self.x)/self.src_len))
            # changed to OB
            return int(len(self.x) / self.src_len)

    '''def __getitem__(self, idx):
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
            output_y).float(), torch.from_numpy(weight).float()'''

    # changed to OB:
    def __getitem__(self, idx):
        if self.isTrain:
            idx = idx * self.step
            output_x = np.stack(self.x[idx:(idx + self.src_len)])
            output_y = np.stack(self.y[idx:(idx + self.src_len)])
        else:
            output_x = np.stack(self.x[idx * self.src_len:(idx * self.src_len + self.src_len)])
            output_y = np.stack(self.y[idx * self.src_len:(idx * self.src_len + self.src_len)])
        if output_y.sum() == 0:
            label = [0, 0, 0]
        else:
            '''center_list = np.where(output_y > 0)[0]
            if len(center_list) > 1:
                center = random.sample(list(center_list), 1)[0]
            else:
                center = center_list[0]
            length = output_y[center]
            label = [1, center / self.src_len, length / self.expression_len]'''
            # todo: changed to only use iou=1
            center_list = np.where(output_y > 0)[0]
            isOne = False
            for c in center_list:
                if c / self.src_len >= 0.4 and c / self.src_len <= 0.6:
                    isOne = True
                    break
            if isOne:
                label = [1, 0.5, 1]
            else:
                label = [0, 0, 0]
        label = torch.tensor(label).float()
        return torch.from_numpy(output_x).permute(0, 3, 1, 2).float(), label


def getDataloader(x, y, isTrain, batch_size, window_length, samples_weight, expression_len, step=None):
    if isTrain:
        dataset = expressionDataset(x, y, True, samples_weight, window_length, step, expression_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        dataset = expressionDataset(x, y, False, samples_weight, window_length, step, expression_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader



