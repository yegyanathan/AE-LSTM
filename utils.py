import torch
import torch.nn as nn
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from loader import get_loader

import pytorch_lightning as pl
from loader import TehranDataset



def train_val_test_split(df, split):

    train_lower_bound = math.floor(split[0] * len(df))
    val_lower_bound = math.floor((split[0] + split[1]) * len(df))

    train  = df[0:train_lower_bound]
    val = df[train_lower_bound:val_lower_bound]
    test = df[val_lower_bound:]

    return train, val, test