import pathlib
import spacy
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import gensim


def load_emoji2vec_dataset(fpath):
    """
    Load the dataset in emoji2vec into pandas DataFrame
    """
    descs = []
    emojis = []
    labels = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            d, e, l = line.strip().split("\t")
            descs.append(d)
            emojis.append(e.strip())    # In neg_samples.txt, all emojis has a space at the end, weird 
            if l == "True":
                labels.append(1)
            else:
                labels.append(0)
    
    return pd.DataFrame({"description": descs, "emoji": emojis, "label": labels})


def load_vocab(fpath):
    """
    Load the emoji vocab.
    """
    vocab = {}
    with open(fpath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            emoji = line.strip()
            vocab[emoji] = idx
    
    return vocab


def transform(sentence: str, max_len, nlp, word2vec):
    """
    Transform a sentence into word vectors. The setence is tokenized into list of words using spacy, then
    the vector of each word is retrieved using the word2vec model.
    
    Args:
        sentence:
            The sentence to transform.
        max_len:
            The maximum length for each sentence.
        nlp:
            The SpaCy model.
        word2vec:
            The gensim word2vec model.
        
    """
    vec_dim = word2vec["hello"].shape[0]
    doc = nlp(sentence)
    vectors = []
    for token in doc:
        if len(vectors) == max_len:
            break
        # If the token isn't in the word2vec model
        if token.text not in word2vec.wv:
            continue
        vectors.append(word2vec[token.text])

    # If the length is < max_len, add padding
    if len(vectors) < max_len:
        num_padding = max_len - len(vectors)
        for i in range(num_padding):
            vectors.append(np.zeros(shape=[vec_dim,]))
    # If the length is too long, we truncate the sentence
    elif len(vectors) > max_len:
        vectors = vectors[:max_len]
    
    assert len(vectors) == max_len

    return np.vstack(vectors)


class EmojiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int, vocab: dict, nlp, word2vec, size=None):
        self.df = df    # The DataFrame containing all samples
        self.max_len = max_len  # Maximum length for the description
        self.vocab = vocab  # Vocab of the emoji
        self.nlp = nlp  # The SpaCy model for tokenization
        self.word2vec = word2vec    # The word2vec model

        if size is None:
            self.size = len(self.df)
        else:
            self.size = size    # Used to display the size of the dataset, since it might be mistakenly inferred (when there're 2 dataloaders for training). 

    def __len__(self):
        # Return the size of the dataset.
        return self.size

    def __getitem__(self, idx):
        #  Fetching a data sample for a given idx.
        
        desc, emoji, label = self.df.iloc[idx]

        emoji_X = np.asarray(self.vocab[emoji])
        desc_X = transform(desc, self.max_len, self.nlp, self.word2vec).astype(np.float32)
        y = np.asarray(label)

        return emoji_X, desc_X, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    current_path = pathlib.Path(__file__).resolve().parent
    project_path = current_path.parent

    vocab = load_vocab(current_path / "data" / "vocab.txt")

    nlp = spacy.load("en_core_web_sm")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(current_path / '..' / 'GoogleNews-vectors-negative300.bin', binary=True)


    train_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "training" / "train.txt")
    train_neg_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "raw_training_data" / "shuf_neg_samples.txt")

    dev_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "training" / "dev.txt")
    test_df = load_emoji2vec_dataset(project_path / "emoji2vec" / "data" / "training" / "test.txt")

    max_len = 5

    train_dataset = EmojiDataset(train_df, max_len, vocab, nlp, word2vec)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_neg_dataset = EmojiDataset(train_neg_df, max_len, vocab, nlp, word2vec)
    train_neg_dataloader = DataLoader(train_neg_dataset, batch_size=64, shuffle=True)
    
    emoji_X, desc_X, y = next(iter(train_dataloader))
    print(emoji_X.shape, desc_X.shape, y.shape)

    emoji_X, desc_X, y = next(iter(train_neg_dataloader))
    print(emoji_X.shape, desc_X.shape, y.shape)
    
    start = time.time()
    iterator = iter(train_dataloader)
    for i in range(10):
        next(iterator)
    end = time.time()

    train_time = end - start

    start = time.time()
    iterator = iter(train_neg_dataloader)
    for i in range(10):
        next(iterator)
    end = time.time()

    train_neg_time = end - start

    print("Time to iterate through 10 batches:")
    print("train_dataset:", train_time)
    print("train_neg_dataset:", train_neg_time)
    # Almost the same time! So using DataFrame.iloc in big DataFrame won't cost performance issue





