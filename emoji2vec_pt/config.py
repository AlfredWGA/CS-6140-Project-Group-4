"""
	This function returns the vocabulary from text data. 
    This function considers only the vocabularies listed on each
    line before an emoji occurs. 
"""
import string
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