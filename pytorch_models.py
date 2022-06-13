from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import EncoderDecoderConfig
import torch.nn as nn
from transformer_encoder.utils import PositionalEncoding
from transformer_encoder import TransformerEncoder


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


class forecastLSTM(nn.Module):

    def __init__(self, 
                    input_size,
                    hidden_size,
                    num_layers,):

        super(forecastLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, 
                                hidden_size, 
                                num_layers, 
                                batch_first = True)

        self.fc = nn.Linear(hidden_size, 1)

        self.init_weights()

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


class AELSTM(nn.Module):


    def __init__(self,
                    input_size, 
                    code_size, 
                    intr_size, 
                    hidden_size, 
                    num_layers,):


        super(AELSTM, self).__init__()

        self.input_size = input_size
        self.code_size = code_size
        self.intr_size = intr_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.AEencoder = Encoder(input_size, code_size, intr_size, 1)
        self.AEdecoder = Decoder(input_size, code_size, intr_size, 1)

        self.forecast = forecastLSTM(code_size,
                                            hidden_size, 
                                            num_layers,)

        
    def forward(self, batch):

        code = self.AEencoder(batch)

        reproduced_X = self.AEdecoder(code)

        prediction = self.forecast(code)

        return prediction, reproduced_X


