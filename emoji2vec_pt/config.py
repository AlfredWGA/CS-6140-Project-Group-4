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
	


preprocess_text("./data/text_desc_data.txt")