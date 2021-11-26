# coding=utf-8
import torch.nn as nn
import torch
import config
import numpy as np


"""
    This class is responsible for the classification of texts 
    into word vectors. 
"""
class TextCNN(nn.Module): 
    def __init__(self, vec_dim): 
        super(TextCNN, self).__init__()
        self.vec_dim = vec_dim 
        self.filter_sizes = [2, 3, 4]
        self.num_filter = 150

        self.relu = nn.ReLU()
        self.conv1d_layers = []

        # Stacking of Convolutional layers in CNN for NLP
        for size in self.filter_sizes: 
            layer = nn.Conv1d(vec_dim, self.num_filter, kernel_size=size)
            nn.init.xavier_normal_(layer.weight.data)
            setattr(self, 'layer_{}'.format(size), layer)
            self.conv1d_layers.append(layer)

        self.fc1 = nn.Linear(self.num_filter * len(self.filter_sizes), 300)


    # Expect embedded vector of size 300. 
    def forward(self, x): 
        """
            x: an embedded vector (batch_size, length, vec_dimension)
        """

        # Change input shape
        # (batch_size, length[row], dim[col]) --> (batch_size, dim[col], length[row])
        x = x.permute(0, 2, 1)

        # Feed into network 
        conv1d_outputs = []
        for layer in self.conv1d_layers:
            out = layer(x)
            activation = self.relu(out)

            # Max pool
            activation, _ = activation.max(dim=2) #????
            conv1d_outputs.append(activation)
  
        x = torch.cat(conv1d_outputs, dim=1)

        # Output 
        out = self.fc1(x)
        return out


data, no_of_words = config.preprocess_text('./data/text_desc_data.txt')
net = TextCNN(vec_dim=300)
x = torch.randn(2, 5, 300)
net.forward(x)
