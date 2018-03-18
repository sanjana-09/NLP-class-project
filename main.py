from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.complexity_classifier import ComplexityClassifier
from utils.scorer import report_score
import nltk

#corpus = nltk.corpus.brown.words()

def execute(language):
    data = Dataset(language)
    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    train_labels = []
    val_labels = []
    max_token_length = 0
    MAX_TOKEN_LENGTH = 4 #11 max token length + 2 baseline features, 24 when pos+baseline+ne


    for sent in data.trainset:
        tokens = len(sent['target_word'].split(' '))
        if tokens > max_token_length:
            max_token_length = tokens

    for sent in data.trainset:
        train_labels.append(sent['gold_label'])
    for sent in data.devset:
        val_labels.append(sent['gold_label'])

    complexity_classifier = ComplexityClassifier(language)
    u_prob = complexity_classifier.read_file('unigram_prob.csv')
    complexity_classifier.set_u_prob(u_prob)
    # u_prob = complexity_classifier.calc_unigram_prob(data.trainset)
    # complexity_classifier.save_to_file(u_prob,'unigram_prob.csv')
    # yjugki

    train_features = complexity_classifier.extract_features(data.trainset,'target_word', MAX_TOKEN_LENGTH, NE = False)
    complexity_classifier.train(train_features, train_labels)

    val_features = complexity_classifier.extract_features(data.devset,'target_word',MAX_TOKEN_LENGTH,NE = False)
    predictions = complexity_classifier.test(val_features)
    report_score(val_labels, predictions, detailed = True)


if __name__ == '__main__':
    execute('english')
    #execute_demo('spanish')

