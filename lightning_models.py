from pyexpat import model
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from pytorch_models import forecastLSTM, AELSTM
from loader import TehranDataset, get_loader
from utils import train_val_test_split, return_based, log_return_based, returns_direction


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
    timestep                -> 100
    k_days                  -> 10

Program args:

    data_path               -> 'Dataset/IKCO1.csv'

"""


class ForecastLSTM(pl.LightningModule):

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
                hidden_layer_size = 64,
                k_days = 30,
                num_layers = 2,
                prob = 0.5,
                timestep = 90,
                ):

        super(ForecastLSTM, self).__init__()

        self.save_hyperparameters()

        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.k_days = k_days
        self.num_layers = num_layers
        self.timestep = timestep
        self.prob = prob


        self.model = forecastLSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    hidden_layer_size = hidden_layer_size,
                                    num_layers = num_layers,
                                    k_days = k_days,
                                    prob = prob,)

        self.loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()



    def setup(self, stage):

        df = pd.read_csv(self.data_path, index_col = [0])
        df = df[:2000]

        df = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]
        df['<CLOSE>_rb'] = pd.DataFrame(return_based(df['<CLOSE>'], 21))
        df['<OPEN>_rb'] = pd.DataFrame(return_based(df['<OPEN>'], 21))
        df['<HIGH>_rb'] = pd.DataFrame(return_based(df['<HIGH>'], 21))
        df['<LOW>_rb'] = pd.DataFrame(return_based(df['<LOW>'], 21))
        df['<VOL>_rb'] = pd.DataFrame(log_return_based(df['<VOL>'], 21))
        df['returns'] = pd.DataFrame(returns_direction(df))

        df = df[21:]

        X = df[['<OPEN>_rb', '<HIGH>_rb', '<LOW>_rb', '<CLOSE>_rb', '<VOL>_rb']]
        y = df[['returns']]

        X_train, X_val, X_test = train_val_test_split(X, self.split)
        y_train, y_val, y_test = train_val_test_split(y, self.split)

        self.scaler_X = MinMaxScaler(feature_range = (0,1)).fit(X_train)
        self.scaler_y = MinMaxScaler(feature_range = (0,1)).fit(y_train)

        X_train_scaled = self.scaler_X.transform(X_train)
        y_train_scaled = self.scaler_y.transform(y_train)

        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)

        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)


        self.train_ds = TehranDataset(X_train_scaled,
                                        y_train_scaled, 
                                        timestep = self.timestep,
                                        k_days = self.k_days)

        self.validation_ds = TehranDataset(X_val_scaled,
                                        y_val_scaled, 
                                        timestep = self.timestep,
                                        k_days = self.k_days)

        self.test_ds = TehranDataset(X_test_scaled,
                                y_test_scaled, 
                                timestep = self.timestep,
                                k_days = self.k_days)


    def on_save_checkpoint(self, checkpoint):

        checkpoint['scaler_X'] = self.scaler_X
        checkpoint['scaler_y'] = self.scaler_y


    def on_load_checkpoint(self, checkpoint):

        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']


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
                                                                            patience = 10),                                                             
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
                hidden_layer_size = 32,
                num_layers = 3,
                prob = 0.21,
                timestep = 100,
                k_days = 10
                ):

        super(AELSTMPredictor, self).__init__()

        self.save_hyperparameters()

        self.data_path = data_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.timestep = timestep
        self.prob = prob
        self.split = split
        self.k_days = k_days


        self.model = AELSTM(input_size = input_size,
                                code_size = code_size,
                                intr_size = intr_size,
                                hidden_size = hidden_size,
                                hidden_layer_size = hidden_layer_size,
                                num_layers = num_layers,
                                prob = prob,
                                k_days = k_days)

        self.loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()
        

    def setup(self, stage):

        df = pd.read_csv(self.data_path, index_col = [0])
        df = df[:2000]

        df = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]
        df['<CLOSE>_rb'] = pd.DataFrame(return_based(df['<CLOSE>'], 21))
        df['<OPEN>_rb'] = pd.DataFrame(return_based(df['<OPEN>'], 21))
        df['<HIGH>_rb'] = pd.DataFrame(return_based(df['<HIGH>'], 21))
        df['<LOW>_rb'] = pd.DataFrame(return_based(df['<LOW>'], 21))
        df['<VOL>_rb'] = pd.DataFrame(log_return_based(df['<VOL>'], 21))
        df['returns'] = pd.DataFrame(returns_direction(df))

        df = df[21:]

        X = df[['<OPEN>_rb', '<HIGH>_rb', '<LOW>_rb', '<CLOSE>_rb', '<VOL>_rb']]
        y = df[['returns']]

        X_train, X_val, X_test = train_val_test_split(X, self.split)
        y_train, y_val, y_test = train_val_test_split(y, self.split)

        self.scaler_X = MinMaxScaler(feature_range = (0,1)).fit(X_train)
        self.scaler_y = MinMaxScaler(feature_range = (0,1)).fit(y_train)

        X_train_scaled = self.scaler_X.transform(X_train)
        y_train_scaled = self.scaler_y.transform(y_train)

        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)

        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)


        self.train_ds = TehranDataset(X_train_scaled,
                                        y_train_scaled, 
                                        timestep = self.timestep,
                                        k_days = self.k_days)

        self.validation_ds = TehranDataset(X_val_scaled,
                                        y_val_scaled, 
                                        timestep = self.timestep,
                                        k_days = self.k_days)

        self.test_ds = TehranDataset(X_test_scaled,
                                y_test_scaled, 
                                timestep = self.timestep,
                                k_days = self.k_days)


    def on_save_checkpoint(self, checkpoint):

        checkpoint['scaler_X'] = self.scaler_X
        checkpoint['scaler_y'] = self.scaler_y


    def on_load_checkpoint(self, checkpoint):

        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']


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
                                                                            patience = 10),                                                             
                    "monitor": "val_loss",
                }
            }

        
    def training_step(self, batch, batch_idx):

        X, y = batch

        outputs, reproduced_x = self.model(X)
        loss1 = self.loss(outputs, y)
        loss2 = self.loss(reproduced_x, X)

        loss = loss1 + loss2

        self.log(name = 'lstm_loss',
            value = loss1,
            on_step = False,
            on_epoch = True,
            prog_bar = False,
            logger = True)

        self.log(name = 'decoder_loss',
                    value = loss2,
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

