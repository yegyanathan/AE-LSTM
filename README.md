# AE-LSTM
Forecasting

Deep Learning-Based Stock Price Prediction 
using Composite LAE-LSTM Model

Classical ML Algorithms take hand-coded features as input and learn an underlying function that 
approximately maps the input to the target. In the case of the stock market, the features could be 
financial markers, and the output could be an entity that is to be predicted. Bestowing the burden 
of selecting the right features for the model upon the engineer may lead to human error and can 
make the forecasting model ineffective.
Representation learning enables the model to learn abstract relations in the data on its own without 
human supervision in feature selection. The representations that are learned by the model are later 
used for approximating the mapping function. Deep Learning (DL) enables the machine to build 
complex representations out of simple concepts by implementing a layer-by-layer approach.
This study focuses on forecasting the maximum closing price in the future k days based on past m 
days of data. For this, the 10 most influential industries in the Tehran Stock Exchange (TSE) were
2
chosen. Seven-year historical data consisting of the Opening value, Highest value, Lowest value, 
and trading volume (OHLCV) of these stocks were used for training the model. The paper proposes 
an architecture that employs LAE for feature selection and stacked LSTM for predicting the closing 
price of the stock.
