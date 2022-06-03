import torch
import torch.nn as nn
import pandas as pd


class Encoder(nn.Module):

    """
    class that encapsulates an encoder
    that tries to learn a crunched representation 
    of the input data.
    
    """

    def __init__(self, input_size, code_size, intr_size, num_layers):

        """
        :param input_size: feature size
        :param code_size: compressed representation size
        :param intermediate_size: size of the intermediate representation
        :param num_layer: number of layers

        """

        super(Encoder, self).__init__()

        self.input_size = input_size
        self.code_size = code_size
        self.intermedite_size = intr_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size, intr_size, num_layers, batch_first = True)
        self.lstm2 = nn.LSTM(intr_size, code_size, num_layers, batch_first = True)

        self.init_weights()


    def init_weights(self):

        """
        Initializes some parameters with values, for easier convergence.
        
        """

        for lstm in list(self.children()):
            for name, param in lstm.named_parameters():
                if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                elif 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)

    def forward(self, X):

        """
        Forward propagation.
        :param X: input tensor, a tensor of dimension (batch_size, timesteps, input_size)
        :return: compressed representation, a tensor of dimension (batch_size, timesteps, code_size)

        """

        intr, _ = self.lstm1(X)
        code, _ = self.lstm2(intr)

        return code


class Decoder(nn.Module):

    """
    class that encapsulates an decoder
    that tries to reproduce the input data from the 
    crunched representation of the input data.
    
    """

    def __init__(self, input_size, code_size, intr_size, num_layers):

        """
        :param input_size: feature size
        :param code_size: compressed representation size
        :param intermediate_size: size of the intermediate representation
        :param num_layer: number of layers
        
        """

        super(Decoder, self).__init__()

        self.input_size = input_size
        self.code_size = code_size
        self.intermedite_size = intr_size
        self.num_layers = num_layers

        self.lstm3 = nn.LSTM(code_size, intr_size, num_layers, batch_first = True)
        self.lstm4 = nn.LSTM(intr_size, input_size, num_layers, batch_first = True)

        self.init_weights()


    def init_weights(self):

        """
        Initializes some parameters with values, for easier convergence.

        """

        for lstm in list(self.children()):
            for name, param in lstm.named_parameters():
                if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                elif 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)


    def forward(self, code):

        """
        Forward propagation.
        :param code: encoded representation
        :return: reproduced input

        """

        intr, _ = self.lstm3(code)
        reproduced_X, _ = self.lstm4(intr)

        return reproduced_X


class PredictNextTimestep(nn.Module):

    """
    Class that encapsulates the pure LSTM architecture.
    :Param input_size: LSTM input size
    :Param hidden_size: LSTM hidden size
    :Param output_layer_size: Fully connected output layer size
    :Param num_layers: LSTM layers
    :Param prob: Dropout probability

    """

    def __init__(self, 
                    input_size,
                    hidden_size,
                    output_layer_size,
                    num_layers,
                    prob):

        super().__init__()

        self.lstm = nn.LSTM(input_size, 
                                hidden_size, 
                                num_layers, 
                                batch_first = True)

        self.fc = nn.Linear(hidden_size, 
                                output_layer_size)

        self.dropout = nn.Dropout(p = prob)

    def init_weights(self):

        """
        
        Initializes the parameters of the model.
        
        """

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

        """
        Param x: tensor of dimension (batch_size, timestep, input_size)
        returns: next timestep of dimension (batch_size, input_size)
        
        """
        
        output, (h,c) = self.lstm(x)
        pred = self.dropout(self.fc(h[-1]))

        return pred


class AELSTM(nn.Module):

    """ 
    Class that encapsulates the AE-LSTM composite architecture.

                ___ AE decoder
               /
    AE encoder 
               \___ Predictor decoder
    
    """

    def __init__(self, input_size, code_size, intr_size, hidden_size, output_layer_size,  num_layers, prob):

        """
        :param input_size: feature size
        :param code_size: compressed representation
        :param intermediate_size: size of the hidden fully connected layer
        :param hidden_size: hidden size of the LSTM
        :param hidden_layer_size: size of fully connected hidden layer
        :param num_layer: number of layers
        :param look_ahead: number of days for forecasting
        
        """

        super(AELSTM, self).__init__()

        self.input_size = input_size
        self.code_size = code_size
        self.intr_size = intr_size
        self.hidden_size = hidden_size
        self.output_layer_size = output_layer_size
        self.num_layers = num_layers
        self.prob = prob

        self.AEencoder = Encoder(input_size, code_size, intr_size, 1)
        self.AEdecoder = Decoder(input_size, code_size, intr_size, 1)

        self.Predictor = PredictNextTimestep(code_size, hidden_size, output_layer_size, num_layers, prob)

        
    def forward(self, X):

        """
        Forward propagation.
        :param X: input tensor, a tensor of dimension (batch_size, timesteps, input_size)
        :return: prediction for the next 'look_ahead' days, reproduced input tensor

        """

        code = self.AEencoder(X)

        reproduced_X = self.AEdecoder(code)

        prediction = self.Predictor(code)

        return prediction, reproduced_X