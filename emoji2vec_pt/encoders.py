# coding=utf-8
import torch.nn as nn
import torch
import config
import numpy as np


class TextBOW(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        """
        x: A batch word vectors with shape [batch_size, max_len, vec_dim]
        """
        return x.mean(dim=1)

"""
    This class is responsible for the classification of texts 
    into word vectors. 
"""
class TextCNN(nn.Module): 
    def __init__(self, data, total_words, vec_dim): 
        super(TextCNN, self).__init__()
        self.vocab_size, self.vec_dim = total_words, vec_dim 
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim=vec_dim,
                                      padding_idx=0, max_norm=5.0)

        self.filter_sizes = [2, 3, 4]
        self.num_filter = 150

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(12, 90, kernel_size=1)
        self.conv2 = nn.Conv1d(90, 150, kernel_size=1)
        self.conv3 = nn.Conv1d(150, 300, kernel_size=1)

        self.pool = nn.MaxPool1d(kernel_size=1) 

        self.fc1 = nn.Linear(10, 30, True) # learns additive bias 
        
        # current issue:
        # the forward pass is taking the whole data, but mapping the 
        # data tensor to a vector of 300 dimenstions. we don't want this. 

        # questions: 
        # If you pool with a size of 1, does it have an effect? 
        # Clarification on dimensions from convolution

    def forward(self, x): 
        x_embed = self.embedding(torch.LongTensor(x)) 
        # For single data input, [12, 300]
        # For all data input, [5584, 12, 300]

        # 12 is for the total length of each sentence with padding
        # 300 is the specified vector dimension
        # not so sure why 5584, our dictionary is < 4000
        # perhaps 5584 is the size with repetition of words. 
        
        x1 = self.conv1(x_embed)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.pool(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.pool(x3)

        flt = torch.flatten(x3)
        print (flt.shape)
        print (x3.shape) # torch.Size([5584, 300, 300])
        

if __name__ == "__main__":
    data, no_of_words = config.preprocess_text('./data/text_desc_data.txt')
    net = TextCNN(data, no_of_words, 300)
    net.forward(data)
