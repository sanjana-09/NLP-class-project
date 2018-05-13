from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.complexity_classifier import ComplexityClassifier
from utils.scorer import report_score
import nltk
import file_io
import spacy

def execute(language):
    data = Dataset(language)
    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    train_labels = []
    val_labels = []
    max_token_length = 0
    MAX_TOKEN_LENGTH = 315 #11 max token length (pos) + 2 baseline + 2 unigram (prob+freq)+ 300 word embeddings 
    #max token length (pos spanish = 10)

    if(language == 'english'):
        WORD_EMB_FILE = 'word_emb.npy'
        U_PROB_FILE = 'unigram_prob_extended.csv'
    else:
        pass

    # for sent in data.trainset:
    #     tokens = len(sent['target_word'].split(' '))
    #     if tokens > max_token_length:
    #         max_token_length = tokens

    for sent in data.trainset:
        train_labels.append(sent['gold_label'])
    for sent in data.devset:
        val_labels.append(sent['gold_label'])

    complexity_classifier = ComplexityClassifier(language)
    # word_vectors_nlp = spacy.load('en_vectors_web_lg')
    # word_emb = file_io.make_word_emb_dict(data.trainset, word_vectors_nlp)
    # file_io.save_dict_npy_file('word_emb.npy', word_emb)
    # print('done')
    # fewi
    #u_prob = complexity_classifier.calc_unigram_freq()
    #complexity_classifier.save_to_file(u_prob,'unigram_freq.csv')
    #print('done')
    #wde

    # sp_u_prob = complexity_classifier.calc_unigram_prob()
    # print('calc done')
    # file_io.save_to_file(sp_u_prob,'spanish_unigram_prob.csv')
    # print('saving done')
    # giug

    nlp = spacy.load('es_core_news_md')
    sp_word_emb = file_io.make_word_emb_dict(data.trainset, nlp)
    print('calc done')
    file_io.save_dict_npy_file(sp_word_emb, 'spanish_word_emb.npy')
    print('saving done')
    iho


    #set files to class 
    # word_emb = file_io.load_npy_file(WORD_EMB_FILE)
    # complexity_classifier.set_word_emb(word_emb)

    # u_prob = file_io.read_file(U_PROB_FILE)
    # complexity_classifier.set_u_prob(u_prob)

    # u_freq = complexity_classifier.read_file('unigram_freq.csv')
    # complexity_classifier.set_u_freq(u_freq)

    def train_and_test(max_token_length = 315, word_emb = False, baseline = False, pos = False,
        unigram_probs = False, syn = False):
        MAX_TOKEN_LENGTH = max_token_length
        print(MAX_TOKEN_LENGTH)
        train_features = complexity_classifier.extract_features(data.trainset,'target_word', MAX_TOKEN_LENGTH, word_emb = word_emb, 
        baseline = baseline, pos = pos, unigram_probs = unigram_probs, syn = syn)

        complexity_classifier.train(train_features, train_labels)
        val_features = complexity_classifier.extract_features(data.devset,'target_word',MAX_TOKEN_LENGTH, 
        word_emb = word_emb, baseline = baseline, pos = pos, unigram_probs = unigram_probs, 
        syn = syn,test = True)

        predictions = complexity_classifier.test(val_features)
        report_score(val_labels, predictions, detailed = True)

    train_and_test(max_token_length = 2,baseline = True)
    #train_and_test(max_token_length = 4, baseline = True, unigram_probs = True, syn = True)
    #train_and_test(max_token_length = 305, baseline = True, unigram_probs = True, word_emb = True, syn = True)
    # train_and_test(max_token_length = 303,baseline = True, word_emb = True, unigram_probs = True)
    #train_and_test(max_token_length = 314, baseline = True, word_emb = True, unigram_probs = True, pos = True)

if __name__ == '__main__':
    execute('spanish')
    #execute_demo('spanish')

