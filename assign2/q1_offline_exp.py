import json
import numpy as np
from math import inf
from time import time
import sys
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import lru_cache
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve, auc
from itertools import cycle
import nltk
nltk.download('stopwords')
nltk.download('punkt')

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
            output_docs.append(_stem(item, ps_stem, en_stop, return_tokens, rem))
        return output_docs
    else:
        return _stem(docs, ps_stem, en_stop, return_tokens, rem)

def get_training_data(json_path, stem=False, bigram=False, limitFeatures=None):
    # loading training data from json
    y, x, n = [], [], []
    if stem:
        for example in json_reader(json_path):
            y.append(int(example['stars']))
            x.append(example['text'])
        x = getStemmedDocuments(x)
        for i in range(len(x)):
            n.append(len(x[i]))
    else:
        for example in json_reader(json_path):
            y.append(int(example['stars']))
            x.append(example['text'].split())
            n.append(len(x[-1]))

    m = len(y)    
    y = np.array(y)
    n = np.array(n)

    if limitFeatures:
      words = {}
      for review in x:
        for word in review:
          if word not in words:
            words[word] = 0
          words[word] += 1
      words = [k for k,v in sorted(words.items(), key=lambda item: item[1])]
      words = set(words[:limitFeatures])
      for i in range(m):
        new_example = []
        for word in x[i]:
          if word in words:
            new_example.append(word)
        x[i] = new_example
        n[i] = len(x[i])

    if bigram:
      for i in range(m):
        for j in range(len(x[i]) - 1):
          x[i][j] += ' ' + x[i][j+1] # convert each word to bigram
        if n[i] > 1:
          x[i].pop() # remove last single word
          n[i] -= 1
    
    vocab = {}
    vocab_count = 1
    for example in x:
        for word in example:
            if word not in vocab:
                vocab[word] = vocab_count
                vocab_count += 1
    d = len(vocab)
    
    for i in range(m):
        for j in range(n[i]):
            x[i][j] = vocab[x[i][j]]  
        x[i] = np.array(x[i])

    return x,y,m,n,d,vocab

def get_naive_bayes_params(x, y, n, m, d, icf=False):
    ec_class = np.zeros(5, dtype='int32') # example count
    wc_class = np.zeros(5, dtype='int32') # word count
    for k in range(5):
        ec_class[k] = np.count_nonzero(1*(y==k+1))
        wc_class[k] = np.sum((y==k+1) * n)

    bag_of_words = np.zeros((5,d+1), dtype='int32') # 5 classes, d vocabulary size
    for i in range(m):
        for w in x[i]: 
            bag_of_words[y[i]-1, w] += 1

    if icf:
      # bag_of_words = bag_of_words / wc_class.reshape((5,1)) # tf
      bag_of_words = bag_of_words * 6 / (1 + np.count_nonzero(bag_of_words, axis=0).reshape((1,d+1))) # icf
      wc_class = np.sum(bag_of_words, axis=1)

    # naive bayes model parameters
    log_phi = np.zeros(5) # 5 classes
    log_theta = np.zeros((5,d+1)) # d parameters for each class

    # finding parameters analytically
    log_phi = np.log(ec_class) - m
    log_theta = np.log(bag_of_words + 1) - np.log(wc_class.reshape((5,1)) + d)

    return log_phi, log_theta

def get_predictions(x, m, log_phi, log_theta):
    predictions = np.zeros(m, dtype='int8')
    for i in range(m):
        chosen_class = 0
        log_prob_class = -inf
        for k in range(5):
            logp = log_phi[k]
            if len(x[i])>0:
                logp += np.sum(log_theta[k,x[i]])
            if logp > log_prob_class:
                log_prob_class = logp
                chosen_class = k + 1
        predictions[i] = chosen_class
    return predictions

def get_score(x, m, log_phi, log_theta):
    probs = np.zeros((m,5))
    for i in range(m):
        for k in range(5):
            logp = log_phi[k]
            if len(x[i])>0:
                logp += np.sum(log_theta[k,x[i]])
            probs[i,k] = logp # denotes log and not actual probability
    # normalising probabilities
    probs = probs - np.max(probs, axis=1).reshape((m,1)) # making largest equal to 0
    probs = np.exp(probs) # converting log of probabilities back to probabilities
    probs = probs / np.sum(probs, axis=1).reshape((m,1)) # making sum of probabilities 1 for each example
    return probs

def get_prediction_accuracy(predictions, y, m):
    correctly_classified = np.count_nonzero(predictions == y)
    return 100*correctly_classified/m
    
def get_test_data(json_path, vocab, stem=False, bigram=False, limitFeatures=None):
    # loading training data from json
    y, x, n = [], [], []
    if stem:
        for example in json_reader(json_path):
            y.append(int(example['stars']))
            x.append(example['text'])
        x = getStemmedDocuments(x)
        for i in range(len(x)):
            n.append(len(x[i]))
    else:
        for example in json_reader(json_path):
            y.append(int(example['stars']))
            x.append(example['text'].split())
            n.append(len(x[-1]))
  
    m = len(y)    
    y = np.array(y)
    n = np.array(n)

    if limitFeatures:
      for i in range(m):
        new_example = []
        for word in x[i]:
          if word in vocab:
            new_example.append(word)
        x[i] = new_example
        n[i] = len(x[i])

    if bigram:
      for i in range(m):
        for j in range(len(x[i]) - 1):
          x[i][j] += ' ' + x[i][j+1] # convert each word to bigram
        if n[i] > 1:
          x[i].pop() # remove last single word
          n[i] -= 1
    
    for i in range(m):
        for j in range(n[i]):
            if x[i][j] in vocab:
                x[i][j] = vocab[x[i][j]]
            else:
                x[i][j] = 0
        x[i] = np.array(x[i])
    
    return x,y,m,n

def get_random_prediction(m):
    predictions = np.random.randint(1,6,m,dtype='int8')
    return predictions

def get_majority_prediction(y):
    max_class, count = 0, 0
    for k in range(1,6):
        c = np.count_nonzero(y==k)
        if c > count:
            max_class = k
            count = c
    return np.array([max_class])

def get_confusion_matrix(predictions, y, m):
    cm = np.zeros((5,5),dtype='int32')
    for i in range(m):
        cm[predictions[i]-1][y[i]-1] += 1
    return cm

def plot_roc(y_test, y_score):
    # Binarise test labels
    y_test = label_binarize(y_test, classes=[1,2,3,4,5])
    n_classes = 5
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for naive bayes text classifier')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    plt.show()

# ---- PART (a) ------
start_time = time()
# Get training data
x_train, y_train, m_train, n_train, d, vocab = get_training_data('./col774_yelp_data/train.json')
print('Vocab size:', d)
# Train naive bayes model
log_phi, log_theta = get_naive_bayes_params(x_train, y_train, n_train, m_train, d)
# Get predictions for training set
predictions_train = get_predictions(x_train, m_train, log_phi, log_theta)
train_accuracy = get_prediction_accuracy(predictions_train, y_train, m_train)
print('Train accuracy:',train_accuracy)
# Get test data
x_test, y_test, m_test, n_test = get_test_data('./col774_yelp_data/test.json', vocab) 
# Get predictions on test set (using naive bayes)
predictions_naive = get_predictions(x_test, m_test, log_phi, log_theta)
test_accuracy = get_prediction_accuracy(predictions_naive, y_test, m_test)
print('Test accuracy (naive bayes model):',test_accuracy)
print('Time taken:', time() - start_time)

# ---- PART (b) -----
start_time = time()
# Get predictions on test set (using random prediction)
predictions_random = get_random_prediction(m_test)
random_accuracy = get_prediction_accuracy(predictions_random, y_test, m_test)
print(f'Test accuracy (random prediction): {random_accuracy}%')
print(f'Improvement of naive bayes model over random prediction: {test_accuracy - random_accuracy}%')
# Get predictions on test set (using majority prediction)
prediction_majority = get_majority_prediction(y_train)
majority_accuracy = get_prediction_accuracy(prediction_majority, y_test, m_test)
print(f'Test accuracy (majority prediction): {majority_accuracy}%')
print(f'Improvement of naive bayes model over majority prediction: {test_accuracy - majority_accuracy}%')
print('Time taken:', time() - start_time)

# ---- PART (c) -----
start_time = time()
# Get confusion matrix for test set (using naive bayes)
confusion_matrix = get_confusion_matrix(predictions_naive, y_test, m_test)
print('Confusion matrix:')
print(confusion_matrix)
print('Time taken:', time() - start_time)

# ---- PART (d) ----
start_time = time()
# Get training data (performing stemming and stopword removal)
x_train_stem, y_train, m_train, n_train_stem, d_stem, vocab_stem = get_training_data('./col774_yelp_data/train.json', stem=True)
print('Vocab size:', d_stem)
# Train naive bayes model (on stemmed data)
log_phi_stem, log_theta_stem = get_naive_bayes_params(x_train_stem, y_train, n_train_stem, m_train, d_stem)
# Get predictions on training set
predictions_train_stem = get_predictions(x_train_stem, m_train, log_phi_stem, log_theta_stem)
train_accuracy_stem = get_prediction_accuracy(predictions_train_stem, y_train, m_train)
print('Train accuracy (Using stemming):',train_accuracy_stem)
# Get test data (performing stemming and stopword removal)
x_test_stem, y_test, m_test, n_test_stem = get_test_data('./col774_yelp_data/test.json', vocab_stem, stem=True) 
# Get predictions on test set
predictions_naive_stem = get_predictions(x_test_stem, m_test, log_phi_stem, log_theta_stem)
test_accuracy_stem = get_prediction_accuracy(predictions_naive_stem, y_test, m_test)
print('Test accuracy (naive bayes model, using stemming):',test_accuracy_stem)
print('Time taken:', time() - start_time)

# ---- PART (f) ----
start_time = time()
# Get probabilities of 5 classes for each example 
y_score = get_score(x_test, m_test, log_phi_stem, log_theta_stem)
# Plot the ROC curves
plot_roc(y_test, y_score)
print('Time taken:', time() - start_time)

# ---- PART (e) ---
# --- Bigram model
start_time = time()
# Get training data
x_train, y_train, m_train, n_train, d, vocab = get_training_data('./col774_yelp_data/train.json', stem=True, bigram=True)
# Train naive bayes model
log_phi, log_theta = get_naive_bayes_params(x_train, y_train, n_train, m_train, d)
# Get predictions for training set
predictions_train = get_predictions(x_train, m_train, log_phi, log_theta)
train_accuracy = get_prediction_accuracy(predictions_train, y_train, m_train)
print('Train accuracy:',train_accuracy)
# Get test data
x_test, y_test, m_test, n_test = get_test_data('./col774_yelp_data/test.json', vocab, stem=True, bigram=True) 
# Get predictions on test set (using naive bayes)
predictions_naive = get_predictions(x_test, m_test, log_phi, log_theta)
test_accuracy = get_prediction_accuracy(predictions_naive, y_test, m_test)
print('Test accuracy (naive bayes model):',test_accuracy)
print('Time taken:', time() - start_time)

# ---- PART (e) ---
# --- Limit features by setting vocabulary to least occuring 100000 words
start_time = time()
# Get training data
x_train, y_train, m_train, n_train, d, vocab = get_training_data('./col774_yelp_data/train.json', stem=True, limitFeatures=100000)
# Train naive bayes model
log_phi, log_theta = get_naive_bayes_params(x_train, y_train, n_train, m_train, d)
# Get predictions for training set
predictions_train = get_predictions(x_train, m_train, log_phi, log_theta)
train_accuracy = get_prediction_accuracy(predictions_train, y_train, m_train)
print('Train accuracy:',train_accuracy)
# Get test data
x_test, y_test, m_test, n_test = get_test_data('./col774_yelp_data/test.json', vocab, stem=True, limitFeatures=100000) 
# Get predictions on test set (using naive bayes)
predictions_naive = get_predictions(x_test, m_test, log_phi, log_theta)
test_accuracy = get_prediction_accuracy(predictions_naive, y_test, m_test)
print('Test accuracy (naive bayes model):',test_accuracy)
print('Time taken:', time() - start_time)

# ---- PART (e) ------
# Normalising using icf (inverse class frequency) to give more weight to words that don't occur in all classes
start_time = time()
# Get training data
x_train, y_train, m_train, n_train, d, vocab = get_training_data('./col774_yelp_data/train.json')
print('Vocab size:', d)
# Train naive bayes model
log_phi, log_theta = get_naive_bayes_params(x_train, y_train, n_train, m_train, d, icf=True)
# Get predictions for training set
predictions_train = get_predictions(x_train, m_train, log_phi, log_theta)
train_accuracy = get_prediction_accuracy(predictions_train, y_train, m_train)
print('Train accuracy:',train_accuracy)
# Get test data
x_test, y_test, m_test, n_test = get_test_data('./col774_yelp_data/test.json', vocab) 
# Get predictions on test set (using naive bayes)
predictions_naive = get_predictions(x_test, m_test, log_phi, log_theta)
test_accuracy = get_prediction_accuracy(predictions_naive, y_test, m_test)
print('Test accuracy (naive bayes model):',test_accuracy)
print('Time taken:', time() - start_time)

