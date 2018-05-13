import copy, csv
import numpy as np
import spacy

def load_npy_file(file_name):
	return np.load(file_name)

def save_dict_npy_file(dictionary, file_name):
	np.save(file_name,dictionary)

def make_word_emb_dict(data, nlp):
	word_emb = {}
	for sent in data:
		doc = nlp(sent['target_word'])
		word_emb[sent['target_word']] = doc.vector
	return word_emb


def calc_unigram_prob(unigram_counts, total_words):
    u_prob = {} #defaultdict
    for word in unigram_counts:
        u_prob[word] = unigram_counts[word]/total_words
    return u_prob

def save_to_file(u_prob,file_name):
    w = csv.writer(open(file_name, "w"))
    for word, prob in u_prob.items():
        w.writerow([word, prob])

def read_file(file_name):
    reader = csv.reader(open(file_name, 'r'))
    u_prob = {}
    for (word,prob) in reader:
        u_prob[word] = float(prob)
    return u_prob