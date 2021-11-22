import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
"""
	This function returns the vocabulary from text data. 
    This function considers only the vocabularies listed on each
    line before an emoji occurs. 
"""
import string, re
def get_vocab(file_path): 
	vocab = 0
	file = open(file_path, "r")
	i = 0
	for line in file: 
		case = line.split(" ")
		if len(case) == 1: 
			vocab += 1
		else: 
			vocab += len(case)
	return vocab

"""
	This function removes the emoji from each line in the dataset. 
"""
def remove_emoji(file_path, dest_path): 
	g = open(dest_path, "w")
	with open(file_path, 'r') as f: 
		for line in f: 
			g.write(line[:-8])
			g.write("\n")

"""
	This function preprocesses the text into a list of indices, 
	where each indice represents the count of the word in the entire
	text. The returned array is padded with 0's. 
"""
def preprocess_text(file_path):
	text = []
	# Clean text: lower case every word and remove 
	with open(file_path, 'r') as f: 
		for line in f:
			line = re.sub(r'[^A-Za-z]+', ' ', line)
			text.append(line.lower())

	# Tokenize words in sentence
	text = [word.split(" ") for word in text]
	clean_text = []
	# Remove '' from tokenized sentences in text 
	for sentence in text: 
		sentence.remove('')
		clean_text.append(sentence)


	# Build vocabulary 
	vocab = {}
	for sentence in clean_text:
		for word in sentence:
			if word in vocab: 
				vocab[word] += 1
				continue
			else: vocab[word] = 0

	# Word to index 
	padded_word_idx = []
	word_idx = []
	for sentence in clean_text: 
		sent = []
		for word in sentence: 
			sent.append(vocab[word])
		word_idx.append(sent)

	# Padding all sentences to match longest sentence length 
	longest_sentence = max([len(sent) for sent in word_idx])
	for sent in word_idx:
		if len(sent) < longest_sentence: 
			pad = [0] * (longest_sentence - len(sent))
			padded_word_idx.append(sent + pad)

	return padded_word_idx

#preprocess_text("./data/text_desc_data.txt")