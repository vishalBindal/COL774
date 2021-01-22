import numpy as np
import cvxopt
from time import time
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from math import inf
import sys

def get_input_all(data):
    y = data[:, -1]  # an array of size m
    # a matrix of size (m, 784), scaling down all values to [0,1]
    x = data[:, :-1] / 255
    m = len(y)  # no of examples
    y = y.reshape((m, 1))  # a matrix of size (m, 1)
    return x, y, m

def get_input(data, class1, class2):
    x, y, m = get_input_all(data)
    # selecting examples with classes class1 and class2
    mb = np.count_nonzero(y == class1) + np.count_nonzero(y == class2)
    yb = np.zeros((mb, 1))
    xb = np.zeros((mb, 784))
    k = 0
    for i in range(m):
        if y[i, 0] == class1 or y[i, 0] == class2:
            if y[i, 0] == class1:
                yb[k, 0] = 1
            else:
                yb[k, 0] = -1
            xb[k, :] = x[i, :]
            k += 1
    return xb, yb, mb

# Gaussian kernel SVM
def get_Km(x, mx, z, mz):
    # |x1-z1|^2 = |x1|^2 + |z1|^2 - sum_i 2*x1i*z1i
    # x : (mx, 784), z: (mz, 784)
    xsq = np.sum(x**2, axis=1).reshape((mx, 1))
    zsq = np.sum(z**2, axis=1).reshape((1, mz))
    xz = np.matmul(x, z.T)
    diff_norm = xsq + zsq - 2*xz  # diff_norm[i,j] = || x[i] - z[j] ||^2
    Km = np.exp(-0.05 * diff_norm)  # Kernel matrix
    return Km

def gaussian_kernel_svm(x, y, m, C_val):
    '''
    Obtain alpha for the gaussian kernel SVM model
    '''
    Km = get_Km(x, m, x, m)  # Kernel matrix

    P = cvxopt.matrix(Km*y*(y.T), tc='d')
    q = cvxopt.matrix(-1 * np.ones((m, 1)), tc='d')
    G = cvxopt.matrix(np.vstack((np.identity(m), -1*np.identity(m))), tc='d')
    h = cvxopt.matrix(np.vstack((C_val*np.ones((m, 1)), np.zeros((m, 1)))), tc='d')
    A = cvxopt.matrix(y.T, tc='d')
    b = cvxopt.matrix(np.zeros(1), tc='d')

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = sol['x']
    return np.array(alpha)

def get_b_gaussian(x, y, m, alpha):
    '''
    Obtain b for the gaussian kernel SVM model
    '''
    Km = get_Km(x, m, x, m)
    wTx = np.sum(alpha.reshape((m, 1))*y*Km, axis=0)
    b1, b2 = inf, -inf
    for i in range(m):
        if y[i, 0] == 1:
            b1 = min(b1, wTx[i])
        else:
            b2 = max(b2, wTx[i])
    b = (b1+b2)*(-1/2)
    return b

# Multi class clasification using self implemented binary classifier
def train_multi_classifier(train_data, C_val):
    '''
    Train the multi-class SVM model by training 45 binary classifiers
    Return alpha and b for all classifiers
    '''
    alphas = {}
    bs = {}
    for class1 in range(10):
        for class2 in range(class1+1, 10):
            # class1 < class2, class1 is assigned y=1
            x, y, m = get_input(train_data, class1, class2)
            alpha = gaussian_kernel_svm(x, y, m, C_val)
            b = get_b_gaussian(x, y, m, alpha)
            alphas[str(class1)+'-'+str(class2)] = alpha
            bs[str(class1)+'-'+str(class2)] = b
    return alphas, bs

def test_multi_class(test_data, train_data, alphas, bs):
    '''
    Test the multi-class model on given test set
    '''
    x, y, m = get_input_all(test_data)
    votes = np.zeros((m, 10))
    scores = np.zeros((m, 10))
    for class1 in range(10):
        for class2 in range(class1+1, 10):
            xtrain, ytrain, mtrain = get_input(train_data, class1, class2)
            Ktest = get_Km(xtrain, mtrain, x, m)

            # get trained params for classifier btw class1 and class2
            alpha = alphas[str(class1)+'-'+str(class2)]
            b = bs[str(class1)+'-'+str(class2)]

            h = np.sum(alpha.reshape((mtrain, 1))*ytrain*Ktest, axis=0)

            for i in range(m):
                if h[i] + b >= 0:
                    votes[i, class1] += 1
                    scores[i, class1] += h[i]
                else:
                    votes[i, class2] += 1
                    scores[i, class2] += h[i]

    predictions = np.zeros(m, dtype='int8')
    for i in range(m):
        max_votes = 0
        chosen_class = 0
        for j in range(10):
            if votes[i, j] > max_votes:
                max_votes = votes[i, j]
                chosen_class = j
            # Resolve ties 
            elif votes[i, j] == max_votes and scores[i, j] > scores[i, chosen_class]:
                chosen_class = j
        predictions[i] = chosen_class

    return predictions

def tuning(train_data):
    x, y, m = get_input_all(train_data)
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.3)
    m_t = y_t.shape[0]
    m_v = y_v.shape[0]
    
    # C values to test
    C_vals = [1, 5, 10]
    best_C, best_accuracy = None, 0

    for C_val in C_vals:
        classifier = svm.SVC(C=C_val, kernel='rbf', gamma=0.05, decision_function_shape='ovo')
        classifier.fit(x_t, y_t.reshape(m_t))
        cv_predictions = classifier.predict(x_v)
        acc = metrics.accuracy_score(y_v.reshape(m_v), cv_predictions)
        if acc > best_accuracy:
            best_C = C_val
            best_accuracy = acc
    return best_C

def run(train_path, test_path):
    # Importing data
    train_data = np.genfromtxt(train_path, delimiter=',')
    test_data = np.genfromtxt(test_path, delimiter=',')
    # Tuning C
    C_val = tuning(train_data)
    # Training multiclass model
    alphas, bs = train_multi_classifier(train_data, C_val)
    # Testing the model on given test data
    test_predictions = test_multi_class(test_data, train_data, alphas, bs)
    return test_predictions

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]
    output = run(train_data, test_data)
    write_predictions(output_file, output)

if __name__ == '__main__':
    main()
