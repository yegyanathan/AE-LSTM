import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from pytorch_models import PredictNextTimestep, AELSTM
from loader import TehranDataset, get_loader
from utils import train_val_test_split

"""
Training args:

    accelerator             -> gpu cpu
    max_epochs              -> 10, 20, 30
    batch_size              -> 32, 64, 128
    callbacks               -> [ModelCheckpoint, EarlyStopping]

Model args:

    input_size              -> 5
    hidden_size             -> 128, 256
    output_layer_size       -> 5
    num_layers              -> 1, 2, 3
    dropout                 -> 0.21, 0.5 
    timestep                -> 10

Program args:

    data_path               -> 'Dataset/IKCO1.csv'

"""


class LSTMPredictor(pl.LightningModule):

    """
    Lightning MOdule for the pure LSTM architecture.
    :Param data_path: file path of the csv file.
    :Param learning_rate: learning rate of the optimizer.
    :Oaram batch_size: batch size of the model.
    :Param input_size: LSTM input size
    :Param hidden_size: LSTM hidden size
    :Param output_layer_size: Fully connected output layer size
    :Param num_layers: LSTM layers
    :Param prob: Dropout probability
    :Param timestep: number of days to look behind.

    """

    def __init__(self,
                data_path,
                split,
                batch_size,
                learning_rate,
                weight_decay,
                input_size = 5,
                hidden_size = 128,
                output_layer_size = 5,
                num_layers = 2,
                prob = 0.21,
                timestep = 10,
                ):

        super(LSTMPredictor, self).__init__()

        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_layer_size = output_layer_size
        self.num_layers = num_layers
        self.timestep = timestep
        self.prob = prob


        self.model = PredictNextTimestep(input_size = input_size,
                                            hidden_size = hidden_size,
                                            output_layer_size = output_layer_size,
                                            num_layers = num_layers,
                                            prob = prob)

        self.loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

        self.scaler_X = MinMaxScaler(feature_range = (0,1))
        self.scaler_y = MinMaxScaler(feature_range = (0,1))

        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()


    def setup(self, stage):

        df = pd.read_csv('Dataset/IKCO1.csv', index_col = [0])

        X = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]
        y = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]

        X_train, X_val, X_test = train_val_test_split(X, self.split)
        y_train, y_val, y_test = train_val_test_split(y, self.split)

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)

        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)


        self.train_ds = TehranDataset(X_train_scaled,
                                        y_train_scaled, 
                                        timestep = self.timestep)

        self.validation_ds = TehranDataset(X_val_scaled,
                                        y_val_scaled, 
                                        timestep = self.timestep)

        self.test_ds = TehranDataset(X_test_scaled,
                                y_test_scaled, 
                                timestep = self.timestep)


    def train_dataloader(self):
        return get_loader(self.train_ds, self.batch_size)


    def val_dataloader(self):
        return get_loader(self.validation_ds, self.batch_size)

    def test_dataloader(self):
        return get_loader(self.test_ds, self.batch_size)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr = self.learning_rate,
                                    weight_decay = self.weight_decay)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                                            mode = 'min',
                                                                            patience = 8),                                                             
                    "monitor": "val_loss",
                }
            }

    
    def forward(self, x):

        pred = self.model(x)

        return pred       


    def training_step(self, batch, batch_idx):

        X, y = batch

        outputs = self.model(X)
        loss = self.loss(outputs, y)

        self.log(name = 'train_loss',
                    value = loss,
                    on_step = False,
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

    def test_step(self, batch, batch_idx):

        X, y = batch

        outputs = self.model(X)

        self.mean_absolute_error(outputs, y)
        self.mean_squared_error(outputs, y)

        self.log(name = 'MAE',
                    value = self.mean_absolute_error,
                    on_epoch = True,
                    logger = True)

        self.log(name = 'MSE',
            value = self.mean_squared_error,
            on_epoch = True,
            logger = True)

        return {'MAE': self.mean_absolute_error,
                    'MSE' : self.mean_squared_error,
                    'log': {'MAE': self.mean_absolute_error, 'MSE' : self.mean_squared_error}}


class AELSTMPredictor(pl.LightningModule):

    """
    Lightning MOdule for the pure LSTM architecture.
    :Param data_path: file path of the csv file.
    :Param learning_rate: learning rate of the optimizer.
    :Oaram batch_size: batch size of the model.
    :Param input_size: Autoencoder input size
    :Param intr_size: Autoencoder input size
    :Param code_size: LSTM input size / Autoencoder code size
    :Param hidden_size: LSTM hidden size
    :Param output_layer_size: Fully connected output layer size
    :Param num_layers: LSTM layers
    :Param prob: Dropout probability
    :Param timestep: number of days to look behind.

    """

    def __init__(self,
                data_path,
                learning_rate,
                weight_decay,
                batch_size,
                split,
                input_size = 5,
                intr_size = 3,
                code_size = 2,
                hidden_size = 128,
                output_layer_size = 5,
                num_layers = 2,
                prob = 0.21,
                timestep = 10,
                ):

        super(AELSTMPredictor, self).__init__()

        self.data_path = data_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_layer_size = output_layer_size
        self.num_layers = num_layers
        self.timestep = timestep
        self.prob = prob
        self.split = split


        self.model = AELSTM(input_size = input_size,
                                code_size = code_size,
                                intr_size = intr_size,
                                hidden_size = hidden_size,
                                output_layer_size = output_layer_size,
                                num_layers = num_layers,
                                prob = prob)

        self.loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

        self.scaler_X = MinMaxScaler(feature_range = (0,1))
        self.scaler_y = MinMaxScaler(feature_range = (0,1))

        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()

    def setup(self, stage):

        df = pd.read_csv(self.data_path, index_col = [0])

        X = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]
        y = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]

        X_train, X_val, X_test = train_val_test_split(X, self.split)
        y_train, y_val, y_test = train_val_test_split(y, self.split)

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)

        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)


        self.train_ds = TehranDataset(X_train_scaled,
                                    y_train_scaled, 
                                    self.timestep)

        self.validation_ds = TehranDataset(X_val_scaled,
                                    y_val_scaled, 
                                    self.timestep)

        self.test_ds = TehranDataset(X_test_scaled,
                                y_test_scaled, 
                                self.timestep)

    def forward(self, x):

        pred, reproduced_x = self.model(x)

        return pred, reproduced_x


    def train_dataloader(self):

        return get_loader(self.train_ds, self.batch_size)

    def val_dataloader(self):

        return get_loader(self.validation_ds, self.batch_size)

    def test_dataloader(self):

        return get_loader(self.test_ds, self.batch_size)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr = self.learning_rate,
                                    weight_decay = self.weight_decay)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                                            mode = 'min',
                                                                            patience = 8),                                                             
                    "monitor": "val_loss",
                }
            }

        


    def training_step(self, batch, batch_idx):

        X, y = batch

        outputs, reproduced_x = self.model(X)
        loss1 = self.loss(outputs, y)
        loss2 = self.loss(reproduced_x, X)

        loss = loss1 + loss2

        self.log(name = 'decoder_loss',
                    value = loss2,
                    on_step = False,
                    on_epoch = True,
                    prog_bar = False,
                    logger = True)

        self.log(name = 'lstm_loss',
                    value = loss1,
                    on_step = False,
                    on_epoch = True,
                    prog_bar = False,
                    logger = True)

        self.log(name = 'train_loss',
                    value = loss,
                    on_step = False,
                    on_epoch = True,
                    prog_bar = True,
                    logger = True)

        return loss


    def validation_step(self, batch, batch_idx):

        X, y = batch

        outputs, reproduced_x = self.model(X)
        loss1 = self.loss(outputs, y)
        loss2 = self.loss(reproduced_x, X)

        loss = loss1 + loss2

        return {'val_loss': loss, 'log': {'val_loss': loss}}


    def validation_epoch_end(self, outputs):

        loss = torch.stack([o['val_loss'] for o in outputs], 0).mean()
        out = {'val_loss': loss}

        self.log(name = 'val_loss',
                    value = loss,
                    on_epoch = True,
                    logger = True)

        return {**out, 'log': out}


    def test_step(self, batch, batch_idx):

        X, y = batch

        outputs, _ = self.model(X)

        MAE = self.mean_absolute_error(outputs, y)
        MSE = self.mean_squared_error(outputs, y)

        self.log(name = 'MAE',
                    value = MAE,
                    on_epoch = True,
                    logger = True)

        self.log(name = 'MSE',
            value = MSE,
            on_epoch = True,
            logger = True)

        return {'MAE': MAE, 'MSE' : MSE, 'log': {'MAE': MAE, 'MSE' : MSE}}

