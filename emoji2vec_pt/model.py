import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from encoders import TextBOW, TextCNN
from torchmetrics import Accuracy


class Emoji2Vec(LightningModule):
    def __init__(self, num_embeddings, embedding_dim, encoder_type="bow"):
        super(Emoji2Vec, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.encoder_type = encoder_type

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.encoder = None
        if encoder_type == "bow":
            self.encoder = TextBOW()
        elif encoder_type == "cnn":
            self.encoder = TextCNN()
        elif encoder_type == "rnn":
            pass
        else:
            raise NotImplementedError(f"The requested encoder {encoder_type} is not implemented.")

        # self.num_classes = 2
        self.criterion = nn.BCEWithLogitsLoss()

        self.acc = Accuracy()

    def forward(self, input_emoji: torch.Tensor, input_word: torch.Tensor):
        """
        input_emoji: A batch of emoji ids, shape [batch_size, 1]
        input_word: A batch of word vectors from the emoji description, shape [batch_size, max_len, embedding_dim]
        """
        emoji_vec = self.embedding(input_emoji) # shape [batch_size, embedding_sim]

        desc_vec = self.encoder(input_word)   # shape [batch_size, embedding_sim]
        
        # shape [batch_size, 1, 1]
        product = torch.bmm(torch.unsqueeze(emoji_vec, 1), torch.unsqueeze(desc_vec, -1))

        # shape [batch_size, 1]
        logits = torch.squeeze(product)

        return logits

    def training_step(self, batch, batch_idx):
        input_emoji = torch.cat([batch[0][0], batch[1][0]])
        input_word = torch.vstack([batch[0][1], batch[1][1]])
        y_true = torch.cat([batch[0][2], batch[1][2]])
        # input_emoji, input_word, y_true = batch

        y_pred = self(input_emoji, input_word)

        loss = self.criterion(y_pred, y_true.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=input_emoji.shape[0])
    
    def validation_step(self, batch, batch_idx):
        input_emoji, input_word, y_true = batch
        y_pred = self(input_emoji, input_word)

        val_loss = self.criterion(y_pred, y_true.float())
        val_acc = self.acc(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True)
        self.log("val_acc", val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        input_emoji, input_word, y_true = batch
        y_pred = self(input_emoji, input_word)

        test_loss = self.criterion(y_pred, y_true.float())
        test_acc = self.acc(y_pred, y_true)
        self.log("test_loss", test_loss, on_epoch=True)
        self.log("test_acc", test_acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer