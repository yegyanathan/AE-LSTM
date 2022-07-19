# Deep Learning-Based Stock Price Prediction using AE-LSTM Model

This study focuses on using Deep Learning for predicting the day ahead closing price based on past stock data. The problem is formulated as a multivariate timeseries forecast. For this, the 10 most influential industries in the Tehran Stock Exchange (TSE) were chosen. Seven-year historical data consisting of the Opening value, Highest value, Lowest value, and trading volume (OHLCV) of these stocks were used for training the model. The Deep Learning architecture integrates an Autoencoder (AE) based on LSTM for feature selection and stacked LSTM for predicting the closing price of the stock.


# Autoencoder

Autoencoder (AE) consists the encoder and the decoder components. The encoder tries to distill the input into a concise representation whereas the decoder tries to reproduce the original input from the concise version. In the process, the model learns to compress the data without losing the relevant information in it. AE usually consists of encoder and decoder that are based on Multi-Layer Perceptrons (MLP). The autoencoder used in this architecture is based on LSTM cell. The design decision was made keeping in mind the temporal nature of the data. The AE takes in 5-dimensional OHLCV data and is trained to compress it into a 2-dimensional representation.



## LSTM

Recurrent Neural Networks (RNN) are a type of Artificial Neural Network that can model sequential/temporal data. As the name suggests, RNNs can capture recurrent relations in a sequence of data with the help of its hidden state which acts like a memory. The hidden state accumulates temporal information and is shared between each timestep. But the RNN suffers from the vanishing gradient problem, which limits it from learning long time dependencies. A variant of the RNN called the LSTM solves this problem by introducing gates that can retain information that is important and discard irrelevant information. It also incorporates a cell state that acts as a highway for efficient gradient flow during Back Propagation Through Time (BPTT). The LSTM Implements 4 gates – Input, Output, Forget, (Input Modulation) Memory Gate. Fig, shows the internal configuration of an LSTM cell.

**![](https://lh5.googleusercontent.com/g8Htzr2wyY5Rbsi1mvkMOQ93JH17BsbeAM7vlGO6hsI5kkNX5BqYbTQ_o6O56nErAFsSbd6KbhaStA9XueFYmpEGoCR98YVrHocDlPyIAdmv4U_l3o7CWVzNUt-B1ojioH8MDX8jD2Zze-STJhI0mIU)**

![alt text](https://www.google.com/imgres?imgurl=https%3A%2F%2Fwikimedia.org%2Fapi%2Frest_v1%2Fmedia%2Fmath%2Frender%2Fsvg%2F1edbece2559479959fe829e9c6657efb380debe7&imgrefurl=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLong_short-term_memory&tbnid=IpIrmxR9j7CTxM&vet=12ahUKEwiszpHhvYX5AhUEgmMGHd41DVUQMygCegUIARDHAQ..i&docid=KUUSnpRkB1yp4M&w=406&h=131&q=lstm%20equations%20&ved=2ahUKEwiszpHhvYX5AhUEgmMGHd41DVUQMygCegUIARDHAQ)

## Model Training
The training phase of the AE-LSTM model is composite in nature. Two separate losses are computed simultaneously at each step, in two separate branches of the model. The Encoder-Decoder branch compresses and decompresses the input data, whereas the Encoder-LSTM branch predicts the closing price of the stock using the compressed representation. The reconstruction loss and the loss due to prediction are backpropagated in their respective branches. The data was split into 80% training set - 10% validation set - 10% testing set. Mean Absolute Error (MAE) was used for calculating the loss. ADAM optimizer was used for minimizing the loss function. Early stopping and weight decaying are some methods that were used to prevent the overfitting of the model.

## Model Validation and Testing:
The validation set was used to monitor the performance of the model. Training loss and validation loss were computed for every epoch to observe the extent to which the model generalized to new data. The effectiveness of the AE-LSTM model was verified by comparing it with a baseline LSTM model. The baseline LSTM model took 16 epochs to converge whereas the AE-LSTM model took 200 epochs to complete training. Hyperparameters were randomly chosen based on intuition.
