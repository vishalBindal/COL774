import json
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from time import time

def json_reader(fname):
    for line in open(fname, mode="r"):
        yield json.loads(line)

def get_data(train_json_path, test_json_path):
    # loading training data from json
    tr_labels, tr_reviews = [], []
    for example in json_reader(train_json_path):
        tr_labels.append(int(example['stars']))
        tr_reviews.append(example['text'])

    vectorizer = CountVectorizer()
    vectorizer.fit(tr_reviews)
    
    x_train = vectorizer.transform(tr_reviews)
    y_train = np.array(tr_labels)

    te_labels, te_reviews = [], []
    for example in json_reader(test_json_path):
        te_labels.append(int(example['stars']))
        te_reviews.append(example['text'])
    
    x_test = vectorizer.transform(te_reviews)
    y_test = np.array(te_labels)

    return x_train, y_train, x_test, y_test

def naive_bayes(x_train, y_train, x_test, y_test):
    start_time = time()
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)
    training_time = time() - start_time
    predictions = classifier.predict(x_test)
    accuracy = 100 * metrics.accuracy_score(y_test, predictions)
    return accuracy, training_time

def svm_liblinear(x_train, y_train, x_test, y_test):
    start_time = time()
    # Tuning C
    x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
    C_values = [0.1, 0.5, 1, 2, 3]
    best_C, best_accuracy = None, 0
    for C in C_values:
        clf = LinearSVC(tol=1e-5, C=C)
        clf.fit(x_t, y_t)
        predictions = clf.predict(x_v)
        accuracy = metrics.accuracy_score(y_v, predictions)
        print(f'Trying C={C}: Accuracy={100*accuracy}%')
        if accuracy > best_accuracy:
            best_C = C
            best_accuracy = accuracy
    tuning_time = time() - start_time
    start_time = time()
    # Training with best C
    classifier = LinearSVC(tol=1e-5, C=best_C)
    classifier.fit(x_train, y_train)
    training_time = time() - start_time
    # Predicting on test set
    predictions = classifier.predict(x_test)
    accuracy = 100*metrics.accuracy_score(y_test, predictions)
    return accuracy, training_time, tuning_time

def svm_sgd(x_train, y_train, x_test, y_test):
    start_time = time()
    # Tuning alpha
    x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
    alpha_values = [1e-4, 1e-5, 1e-6]
    max_iter_values = [500, 1000, 2000]
    best_alpha, best_iter, best_accuracy = None, None, 0
    for alpha in alpha_values:
        for max_iter in max_iter_values:
            clf = SGDClassifier(alpha=alpha, max_iter=max_iter)
            clf.fit(x_t, y_t)
            predictions = clf.predict(x_v)
            accuracy = metrics.accuracy_score(y_v, predictions)
            print(f'Trying alpha={alpha}, max_iter={max_iter}: Accuracy={100*accuracy}%')
            if accuracy > best_accuracy:
                best_alpha = alpha
                best_iter = max_iter
                best_accuracy = accuracy
    tuning_time = time() - start_time
    start_time = time()
    # Training with best alpha and max_iter
    classifier = SGDClassifier(alpha=best_alpha, max_iter=best_iter)
    classifier.fit(x_train, y_train)
    training_time = time() - start_time
    # Predicting on test set
    predictions = classifier.predict(x_test)
    accuracy = 100*metrics.accuracy_score(y_test, predictions)
    return accuracy, training_time, tuning_time

# Importing data
x_train, y_train, x_test, y_test = get_data('./col774_yelp_data/train.json', './col774_yelp_data/test.json')
# Creating tf-idf matrix from count matrix
transformer = TfidfTransformer()
transformer.fit(x_train)
x_train_tfidf, x_test_tfidf = transformer.transform(x_train), transformer.transform(x_test)

# Naive bayes model
print('---Naive bayes---')
naive_bayes_accuracy, naive_bayes_time = naive_bayes(x_train, y_train, x_test, y_test)
print(f'Accuracy: {naive_bayes_accuracy}%')
print(f'Training time: {naive_bayes_time}s')
# SVM (using liblinear)
print('---SVM (Liblinear)---')
svm_liblinear_accuracy, svm_liblinear_time, liblinear_tune_time = svm_liblinear(x_train_tfidf, y_train, x_test_tfidf, y_test)
print(f'Accuracy: {svm_liblinear_accuracy}%')
print(f'Tuning time: {liblinear_tune_time}s')
print(f'Training time: {svm_liblinear_time}s')
# SVM (using SGD)
print('---SVM (SGD)---')
svm_sgd_accuracy, svm_sgd_time, sgd_tune_time = svm_sgd(x_train, y_train, x_test, y_test)
print(f'Accuracy: {svm_sgd_accuracy}%')
print(f'Tuning time: {sgd_tune_time}s')
print(f'Training time: {svm_sgd_time}s')

