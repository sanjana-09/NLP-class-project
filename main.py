from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.complexity_classifier import ComplexityClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils.scorer import report_score
import nltk
import file_io
import spacy
from scipy.sparse import coo_matrix, vstack
from collections import Counter
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def execute(language, test_size = 3328,max_token_length =  303, word_emb = False, unigram_probs = False, baseline = False):
    data = Dataset(language)
    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    train_labels = []
    val_labels = []
    test_labels = []

    max_token_length = 0
    MAX_TOKEN_LENGTH = 315 #11 max token length (pos) + 2 baseline + 2 unigram (prob+freq)+ 300 word embeddings 
    #max token length (pos spanish = 10), sp word embeddings = 50

    if(language == 'english'):
        WORD_EMB_FILE = 'word_emb.npy'
        U_PROB_FILE = 'unigram_prob_extended.csv'
    else:
        WORD_EMB_FILE = 'spanish_word_emb.npy'
        U_PROB_FILE = 'spanish_unigram_prob.csv'


    for sent in data.trainset:
        train_labels.append(sent['gold_label'])
    for sent in data.devset:
        val_labels.append(sent['gold_label'])
    for sent in data.testset:
        test_labels.append(sent['gold_label'])

    complexity_classifier = ComplexityClassifier(language)

    #set files to class 
    word_emb_f = file_io.load_npy_file(WORD_EMB_FILE)
    complexity_classifier.set_word_emb(word_emb_f)

    u_prob = file_io.read_file(U_PROB_FILE)
    complexity_classifier.set_u_prob(u_prob)

    def train_and_test(train_labels, val_labels,max_token_length = 315, word_emb = False, baseline = False, pos = False,
        unigram_probs = False, syn = False, NE = False, test_size = 3328):
        MAX_TOKEN_LENGTH = max_token_length

        train_features, n = complexity_classifier.extract_features(data.trainset,'target_word', MAX_TOKEN_LENGTH, word_emb = word_emb, 
        baseline = baseline, pos = pos, unigram_probs = unigram_probs, syn = syn, NE = NE)

        val_features,n = complexity_classifier.extract_features(data.devset,'target_word',MAX_TOKEN_LENGTH, 
        word_emb = word_emb, baseline = baseline, pos = pos, unigram_probs = unigram_probs, 
        syn = syn, NE = NE, test = True)
        # complexity_classifier.train(train_features, train_labels)
        # predictions = complexity_classifier.test(val_features)
        # print('instance\t','correct label\t','predicted label\t')
        # for i in range(len(predictions)):
        #     if val_labels[i] != predictions[i]:
        #         if len(data.devset[i]['target_word'].split(' ')) > 1:
        #             if val_labels[i] == '0':
        #                 print(data.devset[i]['target_word']+'\t',val_labels[i]+'\t',predictions[i]+'\t')

        # for i in range(len(predictions)):
        #     if val_labels[i] != predictions[i]:
        #         if len(data.devset[i]['target_word'].split(' ')) > 1:
        #             if val_labels[i] =='1':
        #                 print(data.devset[i]['target_word']+'\t',val_labels[i]+'\t',predictions[i]+'\t')
        # gkugk


        # #predictions = complexity_classifier.test(val_features)
        #report_score(val_labels, predictions, detailed = True)

        train_features = vstack([train_features, val_features])
        train_labels+=val_labels
        cv = ShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        return file_io.plot_learning_curve(LogisticRegression(), train_features, train_labels, cv = cv)

    return train_and_test(train_labels,val_labels,max_token_length =  303, word_emb = word_emb, unigram_probs = unigram_probs, baseline = baseline, test_size = test_size)


if __name__ == '__main__':
    #train_sizes_e, train_sizes_e, test_scores_e = execute('spanish', max_token_length = 3, unigram_probs = True, baseline = True)
    train_sizes_e, train_scores_e, test_scores_e = execute('english',max_token_length =  303, word_emb = True, unigram_probs = True, baseline = True)
    train_sizes_s, train_scores_s, test_scores_s = execute('spanish', test_size = 1622,max_token_length =  303, word_emb = True, unigram_probs = True, baseline = True)
    train_sizes_e1, train_scores_e1, test_scores_e1 = execute('english',max_token_length =  3, unigram_probs = True, baseline = True)
    train_sizes_s1, train_scores_s1, test_scores_s1 = execute('spanish', test_size = 1622, max_token_length =  3, unigram_probs = True, baseline = True)

    test_scores_mean_e = np.mean(test_scores_e, axis=1)
    test_scores_mean_s = np.mean(test_scores_s, axis = 1)
    test_scores_mean_e1 = np.mean(test_scores_e1, axis=1)
    test_scores_mean_s1 = np.mean(test_scores_s1, axis = 1)

    plt.figure()
    title = "Training curves"
    plt.title(title)
    ylim=(0.7, 0.85)
    train_sizes = np.linspace(0.1, 1.0, 200)
    plt.ylim(*ylim)
    plt.xlabel("Fraction of training examples")
    plt.ylabel("Macro F1")

    plt.plot(train_sizes, test_scores_mean_e,'r-', label = "English score (unigram+baseline+word emb)")
    plt.plot(train_sizes, test_scores_mean_s, 'g-', label="Spanish score (unigram+baseline+word emb)")
    plt.plot(train_sizes, test_scores_mean_e1,'b-', label = "English score (unigram+baseline)")
    plt.plot(train_sizes, test_scores_mean_s1, 'm-', label="Spanish score (unigram+baseline)")

    plt.legend(loc="best")
    plt.savefig('training2.png')
    plt.show()

