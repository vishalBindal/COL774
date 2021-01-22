import json
import numpy as np
from math import inf
import sys
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import lru_cache

def json_reader(fname):
    for line in open(fname, mode="r"):
        yield json.loads(line)

def _stem(doc, ps_stem, en_stop, return_tokens, rem):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = list(filter(lambda token: token not in en_stop, tokens))
    for i in range(len(stopped_tokens)):
        token = stopped_tokens[i]
        if token not in rem:
            rem[token] = ps_stem(token)
        stopped_tokens[i] = rem[token]
    return stopped_tokens

def getStemmedDocuments(docs, return_tokens=True):
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    ps_stem = lru_cache(maxsize=None)(p_stemmer.stem)
    rem = {}
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(
                _stem(item, ps_stem, en_stop, return_tokens, rem))
        return output_docs
    else:
        return _stem(docs, ps_stem, en_stop, return_tokens, rem)

def get_training_data(json_path):
    # loading training data from json
    y, x, n = [], [], []
    for example in json_reader(json_path):
        y.append(int(example['stars']))
        x.append(example['text'])
    # stemming and stopword removal
    x = getStemmedDocuments(x)
    for i in range(len(x)):
        n.append(len(x[i]))

    m = len(y)
    y = np.array(y)
    n = np.array(n)

    # bigram model
    for i in range(m):
        for j in range(len(x[i]) - 1):
            x[i][j] += ' ' + x[i][j+1]  # convert each word to bigram
        if n[i] > 1:
            x[i].pop()  # remove last single word
            n[i] -= 1

    # build vocabulary
    vocab = {}
    vocab_count = 1
    for example in x:
        for word in example:
            if word not in vocab:
                vocab[word] = vocab_count
                vocab_count += 1
    d = len(vocab)

    # encode word strings into integers
    for i in range(m):
        for j in range(n[i]):
            x[i][j] = vocab[x[i][j]]
        x[i] = np.array(x[i])

    return x, y, m, n, d, vocab


def get_naive_bayes_params(x, y, n, m, d):
    ec_class = np.zeros(5, dtype='int32')  # example count
    wc_class = np.zeros(5, dtype='int32')  # word count
    for k in range(5):
        ec_class[k] = np.count_nonzero(1*(y == k+1))
        wc_class[k] = np.sum((y == k+1) * n)

    # 5 classes, d vocabulary size
    bag_of_words = np.zeros((5, d+1), dtype='int32')
    for i in range(m):
        for w in x[i]:
            bag_of_words[y[i]-1, w] += 1

    # naive bayes model parameters
    log_phi = np.zeros(5)  # 5 classes
    log_theta = np.zeros((5, d+1))  # d parameters for each class

    # finding parameters analytically
    log_phi = np.log(ec_class) - m
    log_theta = np.log(bag_of_words + 1) - np.log(wc_class.reshape((5, 1)) + d)

    return log_phi, log_theta


def get_predictions(x, m, log_phi, log_theta):
    predictions = np.zeros(m, dtype='int8')
    for i in range(m):
        chosen_class = 0
        log_prob_class = -inf
        for k in range(5):
            logp = log_phi[k]
            if len(x[i]) > 0:
                logp += np.sum(log_theta[k, x[i]])
            if logp > log_prob_class:
                log_prob_class = logp
                chosen_class = k + 1
        predictions[i] = chosen_class
    return predictions

def get_test_data(json_path, vocab):
    # loading training data from json
    y, x, n = [], [], []
    for example in json_reader(json_path):
        y.append(int(example['stars']))
        x.append(example['text'])
    # stemming and stopword removal
    x = getStemmedDocuments(x)
    for i in range(len(x)):
        n.append(len(x[i]))

    m = len(y)
    y = np.array(y)
    n = np.array(n)

    # bigram model
    for i in range(m):
        for j in range(len(x[i]) - 1):
            x[i][j] += ' ' + x[i][j+1]  # convert each word to bigram
        if n[i] > 1:
            x[i].pop()  # remove last single word
            n[i] -= 1

    # encode word strings to integers
    for i in range(m):
        for j in range(n[i]):
            if x[i][j] in vocab:
                x[i][j] = vocab[x[i][j]]
            else:
                x[i][j] = 0
        x[i] = np.array(x[i])

    return x, y, m, n

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def run(train_path, test_path):
    x_train, y_train, m_train, n_train, d, vocab = get_training_data(train_path)
    log_phi, log_theta = get_naive_bayes_params(x_train, y_train, n_train, m_train, d)
    x_test, y_test, m_test, n_test = get_test_data(test_path, vocab)
    predictions = get_predictions(x_test, m_test, log_phi, log_theta)
    return predictions

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]
    output = run(train_data, test_data)
    write_predictions(output_file, output)

if __name__ == '__main__':
    main()
