from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import spacy,nltk,csv,math
from nltk.corpus import wordnet
import file_io
from scipy.sparse import coo_matrix, hstack, vstack
import numpy as np

class ComplexityClassifier(object):

    def __init__(self, language):
        self.language = language
        if language == 'english':
            self.avg_word_length = 5.3
            self.corpus_words = nltk.corpus.brown.words()
            self.unigram_counts = Counter(self.corpus_words)
            self.total_words = len(self.corpus_words)
            self.nlp = spacy.load('en')
        else:  
            self.avg_word_length = 6.2
            self.nlp = spacy.load('es')
            self.corpus_words = nltk.corpus.cess_esp.words()
            self.unigram_counts = Counter(self.corpus_words)
            self.total_words = len(self.corpus_words)

        self.model = SVC(class_weight = 'balanced')
        print('svc balanced')
        # self.model = LogisticRegression()
        # print(self.model)

    #save TF-IDF dict to file
    def save_to_file(self,u_prob,file_name):
        w = csv.writer(open(file_name, "w"))
        for word, prob in u_prob.items():
            w.writerow([word, prob])

    def pad_features(self,features, max_token_length):
        for instance_features in features:
            while (len(instance_features) < max_token_length):
                instance_features.append(0.0)
        return features


    def get_unigram_prob(self,target_phrase):
        #print('here')
        prob = 1.0
        words = target_phrase.split(' ')
        for word in words:
            if word in self.u_prob:
                #if prob > self.u_prob[word]:
                prob *= self.u_prob[word] #frq of least frequent word in phrase
            else:
                if(self.language == 'english'):
                    prob *= 8.611840246918683e-07 #lowest prob
                else:
                    prob *= 5.189817577912136e-06

                #prob = 1.0
        return math.log(prob)

    # def get_unigram_freq(self,target_phrase):
    #     #print('here')
    #     freq = 1.0
    #     words = target_phrase.split(' ')
    #     for word in words:
    #         if word in self.u_freq:
    #             if freq > self.u_freq[word]:
    #                 freq = self.u_freq[word] #frq of least frequent word in phrase
    #         else:
    #             #prob *= 8.611840246918683e-07 #lowest prob
    #             freq = 1.0
    #     return math.log(freq)

    def get_word_emb(self,target_phrase):
        return list(self.word_emb[()][target_phrase])

    def calc_unigram_prob(self):
        return file_io.calc_unigram_prob(self.unigram_counts, self.total_words)

    def set_u_prob(self,u_prob):
        self.u_prob = u_prob

    def set_word_emb(self,word_emb):
        self.word_emb = word_emb

    def get_number_syn(self, target_phrase):
        synonyms = []
        senses = []
        words = target_phrase.split(' ')
        for word in words:
            senses += wordnet.synsets(word)
        senses = [str(sense)[8:-2] for sense in senses]
        for sense in senses:
            synonyms += [str(lemma.name()) for lemma in wordnet.synset(sense).lemmas()]
        return len(Counter(synonyms))


    def get_sp_number_syn(self, target_phrase):
        synonyms =[]
        senses = []
        words = target_phrase.split(' ')
        for word in words:
            senses+=wordnet.synsets(word, lang = 'spa')
        senses = [str(sense)[8:-2] for sense in senses]
        for sense in senses:
            synonyms += wordnet.synset(sense).lemma_names('spa')
        return len(Counter(synonyms))


    def extract_features(self,data, key, max_token_length, word_emb = False,
        baseline = True, pos = False, unigram_probs = False,unigram_freq = False, 
        NE = False, syn = False, test = False):
        if baseline:
            #print('here in baseline')
            baseline_features = Baseline(self.language)
            print('Features: Baseline features')
        if NE:
            print('Features: Named entity')
        if pos:
            print('Features: POS')
        if unigram_probs:
            print('Features: Unigram probabilities')
        if unigram_freq:
            print('Features: Unigram frequencies')
        if word_emb:
            print('Features: Word embeddings')
            if test:
                print('word emb test')
                if (self.language == 'english'):
                    word_vectors_nlp = spacy.load('en_vectors_web_lg')
                else:
                    word_vectors_nlp = spacy.load('es_core_news_md')
        if syn:
            print('Features: syn')
        features = []
        n = 0
        i =-1
        print(len(data))
        for sent in data:
            i += 1
            
            target_phrase = sent[key]
            instance_features = []
            #self.get_number_syn(target_phrase)

            if unigram_probs:
                #print(target_phrase)
                instance_features += [self.get_unigram_prob(target_phrase)]

            if unigram_freq:
                #print(target_phrase)
                instance_features += [self.get_unigram_freq(target_phrase)]

            if baseline:
                instance_features += baseline_features.extract_features(target_phrase)

            if word_emb:
                if test:
                    instance_features += list(word_vectors_nlp(target_phrase).vector)
                else:
                    instance_features += self.get_word_emb(target_phrase)
            if syn:
                if(self.language == 'english'):
                    instance_features += [self.get_number_syn(target_phrase)]
                else:
                    instance_features += [self.get_sp_number_syn(target_phrase)]

            if pos:
                instance_features += [token.pos for token in self.nlp(target_phrase)]

            if NE:
                f = [ent.label for ent in self.nlp(target_phrase).ents]
                if not f: #target phrase is not in ents
                    f = [0.0]
                else:
                    n += 1        #jopp
                instance_features += f
            # if i == len(data)-1:
            #     print(n)
            #     upou
            features.append(instance_features)
        features = self.pad_features(features,max_token_length)
        array = np.array(features).reshape(len(features),max_token_length)
        return coo_matrix(array), n
        #return coo_matrix(np.array(features).reshape(len(features),300))

    def train(self, features, labels):
        self.model.fit(features, labels)

    def test(self, features):
        return self.model.predict(features)