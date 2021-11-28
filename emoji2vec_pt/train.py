import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pathlib
import spacy
import gensim
from pytorch_lightning.loggers import WandbLogger

from dataset import EmojiDataset, load_emoji2vec_dataset, load_vocab
from model import Emoji2Vec


current_path = pathlib.Path(__file__).resolve().parent
project_path = current_path.parent

vocab = load_vocab(current_path / "data" / "vocab.txt")

train_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "training" / "train.txt")
train_neg_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "raw_training_data" / "shuf_neg_samples.txt")

dev_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "training" / "dev.txt")
test_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "training" / "test.txt")

print("Loading SpaCy model and pre-trained Word2Vec model...")
nlp = spacy.load("en_core_web_sm")
word2vec = gensim.models.KeyedVectors.load_word2vec_format(project_path / 'GoogleNews-vectors-negative300.bin', binary=True)
print("Loading complete.")

embedding_dim = word2vec["hello"].shape[0]

max_len = 5
batch_size = 64

train_dataset = EmojiDataset(train_df, max_len, vocab, nlp, word2vec)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size // 2, shuffle=True)

train_neg_dataset = EmojiDataset(train_neg_df, max_len, vocab, nlp, word2vec, size=len(train_df))
train_neg_dataloader = DataLoader(train_neg_dataset, batch_size=batch_size // 2, shuffle=True)

val_dataset = EmojiDataset(dev_df, max_len, vocab, nlp, word2vec)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model = Emoji2Vec(num_embeddings=len(vocab), embedding_dim=embedding_dim, encoder_type="bow")

wandb_logger = WandbLogger(project="emoji2vec", save_dir=current_path)

trainer = pl.Trainer(default_root_dir=current_path / "ckpt",
                    callbacks=[EarlyStopping(monitor="val_loss")],
                    gpus=1, # Remove this if training with cpus
                    logger=wandb_logger)
trainer.fit(model, train_dataloaders=[train_dataloader, train_neg_dataloader], val_dataloaders=val_dataloader)