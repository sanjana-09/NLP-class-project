import copy, csv
import numpy as np
import spacy
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

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


def plot_learning_curve(estimator, X, y, ylim=None, cv=None,
                        n_jobs=1, scoring = 'f1_macro',train_sizes=np.linspace(0.1, 1.0, 200)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    return train_sizes, train_scores, test_scores