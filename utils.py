import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller, kpss
import torch



def train_val_test_split(df, split):

    train_lower_bound = math.floor(split[0] * len(df))
    val_lower_bound = math.floor((split[0] + split[1]) * len(df))

    train  = df[0:train_lower_bound]
    val = df[train_lower_bound:val_lower_bound]
    test = df[val_lower_bound:]

    return train, val, test


def create_dataset(X, y, time_step, k_days):

    dataX, dataY = [], []
    
    for i in range(time_step, len(X) - k_days):

        dataX.append(X[i - time_step : i, 0 : 5])
        dataY.append(y[i : i + k_days])
        
    return np.array(dataX), np.array(dataY).squeeze()


def adf_test(timeseries):

    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test statistics', 'p-value', '#lags used', 'Number of observations used'])
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)


def kpss_test(timeseries):

    print ('Results of KPSS Test:')

    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value

    print (kpss_output)



def return_based(timeseries, lookback):

    list = []

    for i in range(21):
        list.append(np.nan)

    for i in range(lookback, len(timeseries)):

        median = timeseries[:i][-lookback:].median()
        scaled = timeseries[i] / median
        list.append(scaled)

    return list


def log_return_based(timeseries, lookback):

    list = []

    timeseries = np.log(timeseries)

    for i in range(21):
        list.append(np.nan)

    for i in range(lookback, len(timeseries)):

        median = timeseries[:i][-lookback:].median()
        sub = timeseries[i] - median
        list.append(sub)

    return list


def returns_direction(df):

    list = []

    for index, row in df.iterrows():
        sub = row['<CLOSE>'] - row['<OPEN>']
        list.append(sub)

    return list