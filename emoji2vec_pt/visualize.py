"""Visualize emoji clusters using TSNE"""

# External dependencies
import sklearn.manifold as man
import matplotlib.pyplot as plt
import pickle as pk
from gensim.models import KeyedVectors
import numpy as np
from pathlib import Path
# from matplotlib.font_manager import FontProperties

# Internal dependencies
#from model import Emoji2Vec
#from parameter_parser import CliParser

project_path = Path(__file__).parent.resolve()

# prop = FontProperties(fname=project_path / 'data\AppleColorEmoji.ttc')


def __visualize():
    # load vectors
    datapath = project_path / 'pre-trained/emoji2vec_bow.bin'
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath, binary=True)
    f = open(project_path / 'data/vocab.txt', encoding='utf-8', errors='ignore')
    vocab = f.read().splitlines()
    #print(len(vocab))
    V = np.zeros(shape=[len(vocab), 300])
    for i, e in enumerate(vocab):
        V[i] = wv_from_bin[e]

    # plot the emoji using TSNE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tsne = man.TSNE(perplexity=50, n_components=2, init='random', n_iter=300000, early_exaggeration=1.0,
                    n_iter_without_progress=1000)
    trans = tsne.fit_transform(V)
    x, y = zip(*trans)
    plt.scatter(x, y, marker='o', alpha=0.0)

    for i in range(len(trans)):
        ax.annotate(vocab[i], xy=trans[i], textcoords='data')

    plt.grid()
    plt.show()

if __name__ == '__main__':
    __visualize()
