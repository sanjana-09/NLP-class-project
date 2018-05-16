from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.complexity_classifier import ComplexityClassifier
from utils.scorer import report_score
import nltk
import file_io
import spacy
from collections import Counter

def execute(language):
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

    # for sent in data.trainset:
    #     tokens = len(sent['target_word'].split(' '))
    #     if tokens > max_token_length:
    #         max_token_length = tokens

    for sent in data.trainset:
        train_labels.append(sent['gold_label'])
    for sent in data.devset:
        val_labels.append(sent['gold_label'])
    for sent in data.testset:
        test_labels.append(sent['gold_label'])

    complexity_classifier = ComplexityClassifier(language)

    #set files to class 
    word_emb = file_io.load_npy_file(WORD_EMB_FILE)
    complexity_classifier.set_word_emb(word_emb)

    u_prob = file_io.read_file(U_PROB_FILE)
    complexity_classifier.set_u_prob(u_prob)

    def train_and_test(max_token_length = 315, word_emb = False, baseline = False, pos = False,
        unigram_probs = False, syn = False, NE = False):
        MAX_TOKEN_LENGTH = max_token_length

        train_features, n = complexity_classifier.extract_features(data.trainset,'target_word', MAX_TOKEN_LENGTH, word_emb = word_emb, 
        baseline = baseline, pos = pos, unigram_probs = unigram_probs, syn = syn, NE = NE)

        complexity_classifier.train(train_features, train_labels)
        # val_features = complexity_classifier.extract_features(data.devset,'target_word',MAX_TOKEN_LENGTH, 
        # word_emb = word_emb, baseline = baseline, pos = pos, unigram_probs = unigram_probs, 
        # syn = syn,test = True)

        test_features,n = complexity_classifier.extract_features(data.devset,'target_word',MAX_TOKEN_LENGTH, 
        word_emb = word_emb, baseline = baseline, pos = pos, unigram_probs = unigram_probs, 
        syn = syn, NE = NE, test = True)
        predictions = complexity_classifier.test(test_features)
        #predictions = complexity_classifier.test(val_features)
        report_score(val_labels, predictions, detailed = True)
        #report_score(test_labels, predictions, detailed = True)

    train_and_test(max_token_length =  303, word_emb = True, unigram_probs = True, baseline = True, synonyms = True)
    # train_and_test(max_token_length = 3, baseline= True, unigram_probs = True)
    # train_and_test(max_token_length = 303, baseline = True, unigram_probs = True, word_emb = True)
    # train_and_test(max_token_length = 304, baseline = True, unigram_probs = True, word_emb = True, syn = True)
    #train_and_test(max_token_length = 315, baseline = True, word_emb = True, unigram_probs = True, syn = True)

if __name__ == '__main__':
    execute('spanish')
    #execute_demo('spanish')        # 
        # misclassified_idx = []
        # for i in range(len(predictions)):
        #     if predictions[i] != test_labels[i]:
        #         misclassified_idx.append(i)
        # print('Target Phrase\t', 'Correct Label\t', 'Predicted Label\t')

        # for idx in misclassified_idx:
        #     if test_labels[idx] =='0':
        #         print(data.testset[idx]['target_word']+'\t',test_labels[idx]+'\t', predictions[idx]+'\t')

        # for idx in misclassified_idx:
        #     if test_labels[idx] == '1': #complex:
        #         print(data.testset[idx]['target_word']+'\t',test_labels[idx]+'\t', predictions[idx]+'\t')
        # jdnhf

