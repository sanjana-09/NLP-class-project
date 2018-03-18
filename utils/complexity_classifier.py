from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from collections import Counter
from sklearn.linear_model import LogisticRegression
import spacy,nltk,csv

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

        self.model = LogisticRegression()

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

    def calc_unigram_prob(self,data):
        u_prob = {} #defaultdict
        unique_words = Counter([word for sent in data for word in sent['target_word'].split(' ')])
        print(len(unique_words))
        for word in unique_words:
            if word in self.corpus_words:
                u_prob[word] = self.unigram_counts[word]/self.total_words
        return u_prob

    def read_file(self,file_name):
        reader = csv.reader(open(file_name, 'r'))
        u_prob = {}
        for (word,prob) in reader:
            u_prob[word] = float(prob)
        return u_prob

    def get_unigram_prob(self,target_phrase):
        prob = 1.0
        for word in target_phrase.split(' '):
            if word in self.corpus_words:
                prob *= self.u_prob[word]
        return prob

    def set_u_prob(self,u_prob):
        self.u_prob = u_prob

    def extract_features(self,data, key, max_token_length, baseline_features = True, pos = True, unigram_probs = True, NE = True):
        if baseline_features:
            baseline = Baseline(self.language)
            print('Features: Baseline features')
        if NE:
            print('Features: Named entity')
        if pos:
            print('Features: POS')
        features = []
        for sent in data:
            target_phrase = sent[key]
            instance_features = []

            if unigram_probs:
                #print(target_phrase)
                instance_features += [self.get_unigram_prob(target_phrase)]
                #print(instance_features)

            if baseline_features:
                instance_features += baseline.extract_features(target_phrase)
            if pos:
                instance_features += [token.pos for token in self.nlp(target_phrase)]

            if NE:
                f = [ent.label for ent in self.nlp(target_phrase).ents]
                if not f: #target phrase is not in ents
                    f = [0.0]
                instance_features += f

            features.append(instance_features)
        return self.pad_features(features, max_token_length)

    def train(self, features, labels):
        self.model.fit(features, labels)

    def test(self, features):
        return self.model.predict(features)