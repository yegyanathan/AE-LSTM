import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller, kpss



def train_val_test_split(df, split):

    train_lower_bound = math.floor(split[0] * len(df))
    val_lower_bound = math.floor((split[0] + split[1]) * len(df))

    train  = df[0:train_lower_bound]
    val = df[train_lower_bound:val_lower_bound]
    test = df[val_lower_bound:]

    return train, val, test


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


def transform_data(df):

    df = df[['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']]

    df['<CLOSE>_p'] = df[['<CLOSE>']].diff(1)
    df['<OPEN>_p'] = df[['<OPEN>']].diff(1)
    df['<HIGH>_p'] = df[['<HIGH>']].diff(1)
    df['<LOW>_p'] = df[['<LOW>']].diff(1)
    df['<VOL>_p'] = np.log(df[['<VOL>']]).diff(1)
    df = df[1:]


    df_X = df[['<OPEN>_p', '<HIGH>_p', '<LOW>_p', '<CLOSE>_p', '<VOL>_p']]
    df_y = df[['<CLOSE>']]

    return df_X, df_y


def create_dataset(X, y, time_step, k_days):

    dataX, dataY = [], []
    
    for i in range(time_step, len(X) - k_days):

        dataX.append(X[i - time_step : i])
        dataY.append(y[i : i + k_days].max())
        
    return np.array(dataX), np.array(dataY).squeeze()