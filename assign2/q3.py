import json
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import sys

def json_reader(fname):
    for line in open(fname, mode="r"):
        yield json.loads(line)

def get_data(train_json_path, test_json_path):
    # loading training data from json
    tr_labels, tr_reviews = [], []
    for example in json_reader(train_json_path):
        tr_labels.append(int(example['stars']))
        tr_reviews.append(example['text'])

    # Creating tf-idf vectors
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tr_reviews)
    x_train = vectorizer.transform(tr_reviews)
    y_train = np.array(tr_labels)

    # loading test data from json
    te_labels, te_reviews = [], []
    for example in json_reader(test_json_path):
        te_labels.append(int(example['stars']))
        te_reviews.append(example['text'])
    
    x_test = vectorizer.transform(te_reviews)
    y_test = np.array(te_labels)

    return x_train, y_train, x_test, y_test

def svm_liblinear(x_train, y_train, x_test, y_test):
    # Tuning C
    x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.3)
    C_values = [0.1, 0.5, 1, 2, 5]
    best_C, best_accuracy = None, 0
    for C in C_values:
        clf = LinearSVC(tol=1e-5, C=C)
        clf.fit(x_t, y_t)
        predictions = clf.predict(x_v)
        accuracy = metrics.accuracy_score(y_v, predictions)
        if accuracy > best_accuracy:
            best_C = C
            best_accuracy = accuracy
    
    # Training with best C
    classifier = LinearSVC(tol=1e-5, C=best_C)
    classifier.fit(x_train, y_train)
    # Predicting on test set
    predictions = classifier.predict(x_test)
    return predictions


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


def run(train_path, test_path):
    x_train, y_train, x_test, y_test = get_data(train_path, test_path)
    predictions = svm_liblinear(x_train, y_train, x_test, y_test)
    return predictions

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]
    output = run(train_data, test_data)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
