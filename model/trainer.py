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

# define the LightningModule
class TransformerLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Multitask_transformer(num_decoder_layers=4, emb_size=576, nhead=4, dim_feedforward=512,
                                           dropout=0.1).float()
        self.loss = nn.functional.mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss,}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('training_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss,}

    def validation_step_end(self, validation_step_outputs):
        if type(validation_step_outputs) == list:
            avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        else:
            avg_loss = validation_step_outputs['loss']
        self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer



class expressionDataset(Dataset):
    def __init__(self, x, y, isTrain, src_len=512):
        self.x = x
        self.y = y
        self.src_len = src_len
        self.isTrain = isTrain

    def __len__(self):
        if self.isTrain:
            return len(self.x) - self.src_len #todo: testing
            #return 4
        else:
            return int(np.ceil(len(self.x)/self.src_len))

    def __getitem__(self, idx):
        if self.isTrain:
            output_x = np.stack(self.x[idx:(idx+self.src_len)])
            output_y = np.stack(self.y[idx:(idx+self.src_len)])
        else:
            output_x = np.stack(self.x[idx*self.src_len:(idx*self.src_len + self.src_len)])
            output_y = np.stack(self.y[idx*self.src_len:(idx*self.src_len + self.src_len)])

        return torch.from_numpy(output_x).permute(0,3,1,2).float(), torch.from_numpy(output_y).float()


def getDataloader(x, y, isTrain, batch_size):
    if isTrain:
        dataset = expressionDataset(x, y, True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = expressionDataset(x, y, False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def getCallbacks():
    checkpoint_callback = ModelCheckpoint(monitor='validation_loss_each_epoch', save_last=True, save_top_k=1,
                                          mode='min')
    early_stop_callback = EarlyStopping(
        monitor='validation_loss_each_epoch', min_delta=0.00, patience=10, verbose=False, mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return [lr_monitor, checkpoint_callback, early_stop_callback]


