from numpy.core import numeric
import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class Emoji2Vec(LightningModule):
    def __init__(self, num_embeddings, embedding_dim, encoder="bow"):
        super(Emoji2Vec, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.encoder = encoder

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        if encoder == "bow":
            pass
        elif encoder == "cnn":
            pass
        elif encoder == "rnn":
            pass
        else:
            raise NotImplementedError(f"The requested encoder {encoder} is not implemented.")

        self.num_classes = 2
        self.criterion = nn.BCEWithLogitsLoss()
        # self.filter_sizes = [2, 3, 4]
        # self.num_filter = 150

        # self.conv1d_layers = []
        # for size in self.filter_sizes:
        #     conv1d = nn.Conv1d(vec_dim, self.num_filter, kernel_size=size)
        #     nn.init.xavier_normal_(conv1d.weight.data)
        #     setattr(self, "conv1d_{}".format(size), conv1d)
        #     self.conv1d_layers.append(conv1d)

        # self.relu = nn.ReLU()
        # # self.global_maxpool = lambda x: torch.max(x, dim=2)

        # self.fc1 = nn.Linear(self.num_filter * len(self.filter_sizes), 256)
        # self.dropout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, input_emoji: torch.Tensor, input_word: torch.Tensor):
        """
        input_emoji: A batch of emoji ids, shape [batch_size, 1]
        input_word: A batch of word vectors from the emoji description, shape [batch_size, max_len, embedding_dim]
        """
        emoji_vec = self.embedding(input_emoji) # shape [batch_size, embedding_sim]

        # desc_vec = torch.mean(input_word, dim=1)   # shape [batch_size, embedding_sim]
        desc_vec = TextRNN(input_word)
        
        # shape [batch_size, 1, 1]
        product = torch.bmm(torch.unsqueeze(emoji_vec, 1), torch.unsqueeze(desc_vec, -1))

        logits = torch.squeeze(product)

        return logits
        # (batch, vec_dim, length)
        # x = x.permute(0, 2, 1)

        # conv1d_outputs = []
        # for conv1d in self.conv1d_layers:
        #     o = self.relu(conv1d(x))
        #     # Global max pooling
        #     # (batch, num_filter)
        #     o, _ = o.max(dim=2)
        #     conv1d_outputs.append(o)

        # x = torch.cat(conv1d_outputs, dim=1)

        # x = self.dropout(self.fc1(x))
        # x = self.fc2(x)
        # return x

    def training_step(self, batch, batch_idx):
        input_emoji, input_word, y_true = batch
        y_pred = self(input_emoji, input_word)

        loss = F.cross_entropy(y_pred, y_true)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss
    
    def training_epoch_end(self, training_step_outputs)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer