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
		

net = TextCNN('./data/train.txt', 300)
