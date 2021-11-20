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
	def __init__(self, data_path, vec_dim): 
		super(TextCNN, self).__init__()
		vocab_size, vec_dim = config.get_vocab(data_path), vec_dim
		self.embedding = nn.Embedding.from_pretrained(vocab_size,
                                                      embedding_dim=vec_dim,
                                                      padding_idx=0, 
                                                      max_norm=5.0)

        self.filter_sizes = [2, 3, 4]
        self.num_filter = 150

        self.conv1d_layers = []
        ### FROM HERE NOT ENTIRELY SURE ABOUT THE ARCHITECTURE
        for size in self.filter_sizes:
            conv1d = nn.Conv1d(vec_dim, self.num_filter, kernel_size=size)
            nn.init.xavier_normal_(conv1d.weight.data)
            setattr(self, 'conv1d_{}'.format(size), conv1d)
            self.conv1d_layers.append(conv1d)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.num_filter * len(self.filter_sizes), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, vec_dim)

net = TextCNN('./data/train.txt', 300)
