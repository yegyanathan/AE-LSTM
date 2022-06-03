import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


class TehranDataset(Dataset):

    """
    Tehran Stock Exchange (TSE) dataset
    :Param X_scaled: normalized x
    :Param y_scaled: normalized y
    :Param timestep: number of days to look behind.
    
    """

    features = ['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']
    
    def __init__(self, 
                    X_scaled,
                    y_scaled, 
                    timestep):
        
        self.X_scaled = X_scaled
        self.y_scaled = y_scaled
        self.timestep = timestep

        self.data_X, self.data_Y = TehranDataset.create_dataset(X_scaled,
                                                                    y_scaled,
                                                                    timestep)
        

    @staticmethod
    def create_dataset(X, y, time_step = 1):

        dataX, dataY = [], []
        
        for i in range(time_step, len(X)):

            dataX.append(X[i - time_step : i, 0 : 5])
            dataY.append(y[i, 0 : 5])
            
        return np.array(dataX), np.array(dataY)
    

    def __len__(self):
        
        return len(self.data_X)


    def __getitem__(self, index):

        timestep_X = self.data_X[index]
        timestep_y = self.data_Y[index]

        timestep_X = torch.tensor(timestep_X)
        timestep_y = torch.tensor(timestep_y)

        return timestep_X.float(), timestep_y.float()




def get_loader(dataset, batch_size, num_workers = 2, shuffle = False, drop_last = True):

    loader = DataLoader(dataset = dataset,
                         batch_size = batch_size, 
                         num_workers = num_workers,
                         shuffle = shuffle, 
                         drop_last = drop_last)

    return loader