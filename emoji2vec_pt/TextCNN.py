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
		self.vocab_size, self.vec_dim = config.get_vocab(data_path), vec_dim
		self.embedding = nn.Embedding(self.vocab_size, embedding_dim=vec_dim,
									  padding_idx=0, max_norm=5.0)

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

	def forward(self, x): 
		x_embed = self.embedding(x)
		print (x_embed)
		# x_reshape = x_embed.permute(0, self.vec_dim, 1) 
		# x_conv = []
		# # Conv layers
		# for layer in self.conv1d_layers: 
		# 	x_conv.append(self.relu(layer(x_reshape)))
		# Pooling layers 
		# for layer in self.conv1d_layers: 
		# 	nn.MaxPool2d(layer, kernel_size=4)

		print (x_conv)


net = TextCNN('./data/train.txt', 300)
net.forward("We would live long")
