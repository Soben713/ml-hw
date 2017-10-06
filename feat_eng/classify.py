import argparse
import string
from csv import DictReader, DictWriter

import numpy as np
import random
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.classification import accuracy_score

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

SEED = 1152

FEATURE_STR_DIVIDER = '--'


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), stop_words=stopwords.words('english'))

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))


def debug_accuracy(classifier, x, y, examples):
    predictions = classifier.predict(x)

    errors = []

    false_positive = 0
    false_negative = 0

    for i in range(len(list(x))):
        if predictions[i] != y[i]:
            errors.append((examples[i], y[i], predictions[i]))
            if predictions[i] == 1:
                false_positive += 1
            else:
                false_negative += 1

    for i in range(50):
        print("True Label: %s, Prediction: %s, Data: %s, \t Original Data: %s" % (
            errors[i][1], errors[i][2], add_features(errors[i][0]), errors[i][0]))

    print("Accuracy: %f" % accuracy_score(y, predictions))

    print("False positive: %s \t False negative: %s" % (false_positive, false_negative))


def add_n_grams(sentence):
    char = FEATURE_STR_DIVIDER
    for ngram_length in range(2, 3):
        char += ' ' + " ".join("".join(cc for cc in x)
                               for x in ngrams(sentence.split(), ngram_length))
    sentence += char
    return sentence


def add_if_has_episode(sentence):
    episode_occurances = re.findall('\d+x\d', sentence)
    sentence += " HASEPISODE%s" % len(episode_occurances)
    # sentence += len(episode_occurances) * " HASEPISODE" DOES NOT WORK
    return sentence


def add_puncs(sentence):
    occurances = re.findall(r'[!?"]', sentence)

    for punc in occurances:
        sentence += " punc%s" % string.punctuation.find(punc)
    # sentence += " QUOTEsNUM:%s" % len(quotes_occurances)
    return sentence


def lemmatize(sentence):
    lmtzr = WordNetLemmatizer()

    updated_sentence = ''

    for token in sentence.split():
        updated_sentence += lmtzr.lemmatize(lmtzr.lemmatize(token, 'v')) + ' '

    return updated_sentence


def add_trope(sentence, trope):
    prefix = 'trope'
    words = re.findall('[A-Z][^A-Z]*', trope)
    words = [prefix + w for w in words]
    sentence += ' ' + ' '.join(words)
    return sentence


def remove_puncs(sentence):
    return re.sub(r'[!-"?,.]', '', sentence)


def add_features(x):
    sentence = x[kTEXT_FIELD]

    # sentence = add_n_grams(sentence)

    sentence = add_trope(sentence, x['trope'])

    # sentence += ' trope' + x['trope']
    # sentence += ' page' + x['page']

    sentence = add_if_has_episode(sentence)

    # sentence.translate(None, string.punctuation)
    sentence = add_puncs(sentence)

    # Lemmatize has to be here since we get rid of punctuations and they are needed in some of the previous features
    sentence = remove_puncs(sentence)
    sentence = lemmatize(sentence)

    return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--debug', default=0, type=int,
                        help="test on half of the training")

    flags = parser.parse_args()

    if flags.debug == 1:
        print("DEBUG DO NOT SUBMIT!")

    # Cast to list to keep it all in memory
    train_raw = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    random.Random(SEED).shuffle(train_raw)

    if flags.debug == -1:
        train = train_raw
        test_train = list()
    else:
        m = int((9 / 10) * len(train_raw))
        train = train_raw[:m]
        test_train = train_raw[m:]

    feat = Featurizer()

    labels = ['False', 'True']
    # for line in train:
    #     if not line[kTARGET_FIELD] in labels:
    #         labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))
    x_train = feat.train_feature(add_features(x) for x in train)
    x_test = feat.test_feature(add_features(x) for x in test)

    # pages = set()
    # # for x in test:
    # #     pages.add(x['page'])
    # for x in train:
    #     pages.add(x['page'])
    # print(pages)

    if flags.debug:
        x_test_train = feat.test_feature(add_features(x) for x in test_train)

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    y_test_train = array(list(labels.index(x[kTARGET_FIELD])
                              for x in test_train))

    print(len(train), len(y_train))
    print(set(y_train))

    if flags.debug:
        print("DEBUG:", len(test_train), len(y_test_train))

    # Train classifier
    if flags.debug:
        lr = SGDClassifier(loss='log', penalty='l2', shuffle=False)
    else:
        lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    if flags.debug:
        debug_accuracy(lr, x_test_train, y_test_train, test_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
