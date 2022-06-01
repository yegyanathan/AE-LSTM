import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from loader import get_loader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

from loader import TehranDataset

"""
Training args:

    accelerator
    devices
    auto_lr_find
    auto_scale_batch_size
    max_epochs
    callbacks = [ModelCheckpoint]

Model args:

    input_size
    hidden_size
    hidden_layer_size
    num_layers
    look_ahead

Program args:

    data_path

"""

class PredictNextTimestep(nn.Module):

    """
    Desc

    """

    def __init__(self, 
                    input_size = 5,
                    hidden_size = 128,
                    output_size = 5,
                    num_layers = 2,):

        super().__init__()

        self.lstm = nn.LSTM(input_size, 
                                hidden_size, 
                                num_layers, 
                                batch_first = True)

        self.fc = nn.Linear(hidden_size, 
                                output_size)

    def init_weights(self):

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

        nn.init.kaiming_uniform_(self.fc.weight.data)
        nn.init.constant_(self.fc.weight.data, 0)

    def forward(self, x):
        
        output, (h,c) = self.lstm(x)
        pred = self.fc(h[-1])

        return pred


class LSTMPredictor(pl.LightningModule):

    """
    Desc

    """

    def __init__(self,
                data_path,
                learning_rate,
                batch_size,
                split,
                input_size = 5,
                hidden_size = 128,
                output_size = 5,
                num_layers = 2,
                timestep = 10,
                ):

        super(LSTMPredictor, self).__init__()

        self.data_path = data_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer_size = output_size
        self.num_layers = num_layers
        self.timestep = timestep
        self.split = split

        self.train_ds = None
        self.validation_ds = None

        self.model = PredictNextTimestep(input_size = 5,
                                            hidden_size = 128,
                                            output_size = 5,
                                            num_layers = 2,)

        self.loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

        self.scaler_X = MinMaxScaler(feature_range = (0,1))
        self.scaler_y = MinMaxScaler(feature_range = (0,1))

    def prepare_data(self):

        df = pd.read_csv(self.data_path, index_col = [0])

        X = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]
        y = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]

        def _train_dev_split(df, split):

            train_lower_bound = math.floor(split[0] * len(df))
            dev_lower_bound = math.floor((split[0] + split[1]) * len(df))

            train  = df[0:train_lower_bound]
            dev = df[train_lower_bound:dev_lower_bound]

            return train, dev

        X_train, X_val = _train_dev_split(X, self.split)
        y_train, y_val = _train_dev_split(y, self.split)

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)


        self.train_ds = TehranDataset(X_train_scaled,
                                     y_train_scaled, 
                                     self.timestep)

        self.validation_ds = TehranDataset(X_val_scaled,
                                     y_val_scaled, 
                                     self.timestep)


    def train_dataloader(self):

        return get_loader(self.train_ds)

    def val_dataloader(self):

        return get_loader(self.validation_ds)

    def forward(self, x):

        pred = self.model(x)

        return pred

    def configure_optimizers(self):

        return torch.optim.Adam(self.model.parameters(),
                                    lr = self.learning_rate)


    def training_step(self, batch, batch_idx):

        X, y = batch

        outputs = self.model(X)
        loss = self.loss(outputs, y)

        self.log(name = 'train_loss',
                    value = loss,
                    on_epoch = True,
                    prog_bar = True,
                    logger = True)

        return loss


    def validation_step(self, batch, batch_idx):

        X, y = batch

        outputs = self.model(X)
        loss = self.loss(outputs, y)

        return {'val_loss': loss, 'log': {'val_loss': loss}}


    def validation_epoch_end(self, outputs):

        loss = torch.stack([o['val_loss'] for o in outputs], 0).mean()
        out = {'val_loss': loss}

        self.log(name = 'val_loss',
                    value = loss,
                    on_epoch = True,
                    logger = True)

        return {**out, 'log': out}
