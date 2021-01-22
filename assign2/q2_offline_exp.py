import numpy as np
import cvxopt
from time import time
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from math import inf

# Importing data
train_data = np.genfromtxt('./fashion_mnist/train.csv', delimiter=',')
val_data = np.genfromtxt('./fashion_mnist/val.csv', delimiter=',')
test_data = np.genfromtxt('./fashion_mnist/test.csv', delimiter=',')

def get_input_all(data):
  y = data[:,-1] # an array of size m 
  x = data[:,:-1] / 255 # a matrix of size (m, 784), scaling down all values to [0,1]
  m = len(y) # no of examples
  y = y.reshape((m,1)) # a matrix of size (m, 1)
  return x, y, m

def get_input(data, class1, class2):
  x, y, m = get_input_all(data)
  # selecting examples with classes class1 and class2
  mb = np.count_nonzero(y==class1) + np.count_nonzero(y==class2)
  yb = np.zeros((mb,1))
  xb = np.zeros((mb,784))
  k = 0
  for i in range(m):
    if y[i,0] == class1 or y[i,0] == class2:
      if y[i,0] == class1:
        yb[k,0] = 1
      else:
        yb[k,0] = -1
      xb[k,:] = x[i,:]
      k += 1
  return xb,yb,mb

# Linear SVM
def linear_svm(x, y, m):
  P = cvxopt.matrix(np.matmul(x*y, (x*y).T), tc='d') # P_ij = y(i) . y(j) . x(i)T . x(j)
  q = cvxopt.matrix(-1 * np.ones((m,1)), tc='d')
  G = cvxopt.matrix(np.vstack((np.identity(m), -1*np.identity(m))), tc='d')
  h = cvxopt.matrix(np.vstack((np.ones((m,1)), np.zeros((m,1)))), tc='d')
  A = cvxopt.matrix(y.T, tc='d')
  b = cvxopt.matrix(np.zeros(1), tc='d')
  
  sol = cvxopt.solvers.qp(P, q, G, h, A, b)
  alpha = sol['x']
  return np.array(alpha)

def get_support_vectors(alpha, x, y, m, threshold=1e-5):
  new_alphas, support_vectors, new_ys = [], [], []
  for i in range(m):
    if alpha[i] > threshold:
      support_vectors.append(x[i,:])
      new_alphas.append(alpha[i])
      new_ys.append(y[i,0])
  new_m = len(new_ys)
  return np.array(support_vectors), np.array(new_ys).reshape((new_m,1)), new_m, np.array(new_alphas)

def get_params(x, y, m, alpha):
  w = np.sum(alpha*y*x, axis=0)
  b1, b2 = inf, -inf
  for i in range(m):
    if y[i,0] == 1:
      b1 = min(b1, np.dot(w, x[i,:]))
    else:
      b2 = max(b2, np.dot(w, x[i,:]))
  b = (b1+b2)*(-1/2)
  return w, b

def get_accuracy_linear(w, b, m, x, y):
  predictions = np.zeros(m, dtype='int8')
  correctly_classified = 0
  for i in range(m):
    if np.dot(w, x[i,:]) + b >= 0:
      correctly_classified += (y[i,0] == 1)
      predictions[i] = 1
    else:
      correctly_classified += (y[i,0] == -1)
      predictions[i] = -1
  prediction_accuracy = correctly_classified * 100 / m
  return prediction_accuracy, predictions

# Gaussian kernel SVM
def get_Km(x, mx, z, mz):
  # |x1-z1|^2 = |x1|^2 + |z1|^2 - sum_i 2*x1i*z1i
  # x : (mx, 784), z: (mz, 784)
  xsq = np.sum(x**2, axis=1).reshape((mx, 1))
  zsq = np.sum(z**2, axis=1).reshape((1,mz))
  xz = np.matmul(x, z.T)
  diff_norm = xsq + zsq - 2*xz # diff_norm[i,j] = || x[i] - z[j] ||^2
  Km = np.exp(-0.05 * diff_norm) # Kernel matrix
  return Km

def gaussian_kernel_svm(x, y, m):
  Km = get_Km(x, m, x, m) # Kernel matrix
  
  P = cvxopt.matrix(Km*y*(y.T), tc='d')
  q = cvxopt.matrix(-1 * np.ones((m,1)), tc='d')
  G = cvxopt.matrix(np.vstack((np.identity(m), -1*np.identity(m))), tc='d')
  h = cvxopt.matrix(np.vstack((np.ones((m,1)), np.zeros((m,1)))), tc='d')
  A = cvxopt.matrix(y.T, tc='d')
  b = cvxopt.matrix(np.zeros(1), tc='d')

  sol = cvxopt.solvers.qp(P, q, G, h, A, b)
  alpha = sol['x']
  return np.array(alpha)

def get_b_gaussian(x, y, m, alpha):
  Km = get_Km(x, m, x, m)
  wTx = np.sum(alpha.reshape((m,1))*y*Km, axis=0)
  b1, b2 = inf, -inf
  for i in range(m):
    if y[i,0] == 1:
      b1 = min(b1, wTx[i])
    else:
      b2 = max(b2, wTx[i])
  b = (b1+b2)*(-1/2)
  return b

def get_accuracy_gaussian(alpha, b, m, x, y, x_sup, m_sup, y_sup):
  predictions = np.zeros(m, dtype='int8')
  Ktest = get_Km(x_sup, m_sup, x, m)
  h = np.sum(alpha.reshape((m_sup,1))*y_sup*Ktest, axis=0)
  correctly_classified = 0
  for i in range(m):
    if h[i] + b >= 0:
      correctly_classified += (y[i,0] == 1)
      predictions[i] = 1
    else:
      correctly_classified += (y[i,0] == -1)
      predictions[i] = -1
  prediction_accuracy = correctly_classified * 100 / m
  return prediction_accuracy, predictions

# Multi class clasification using self implemented binary classifier
def train_multi_classifier(train_data):
  alphas = {} 
  bs = {}
  for class1 in range(10):
    for class2 in range(class1+1,10):
      # class1 < class2, class1 is assigned y=1
      x, y, m = get_input(train_data, class1, class2)
      alpha = gaussian_kernel_svm(x, y, m)
      b = get_b_gaussian(x, y, m, alpha)
      alphas[str(class1)+'-'+str(class2)] = alpha
      bs[str(class1)+'-'+str(class2)] = b 
  return alphas, bs

def test_multi_class(test_data, train_data, alphas, bs):
  x, y, m = get_input_all(test_data)
  votes = np.zeros((m,10))
  scores = np.zeros((m,10))
  for class1 in range(10):
    for class2 in range(class1+1, 10):
      xtrain, ytrain, mtrain = get_input(train_data, class1, class2)
      Ktest = get_Km(xtrain, mtrain, x, m)

      # get trained params for classifier btw class1 and class2
      alpha = alphas[str(class1)+'-'+str(class2)]
      b = bs[str(class1)+'-'+str(class2)]
      
      h = np.sum(alpha.reshape((mtrain,1))*ytrain*Ktest, axis=0)
      
      for i in range(m):
        if h[i] + b >= 0:
          votes[i,class1] += 1
          scores[i,class1] += h[i]
        else:
          votes[i,class2] += 1
          scores[i,class2] += h[i]
  
  predictions = np.zeros(m, dtype='int8')
  for i in range(m):
    max_votes = 0
    chosen_class = 0
    for j in range(10):
      if votes[i,j] > max_votes:
        max_votes = votes[i,j]
        chosen_class = j
      elif votes[i,j] == max_votes and scores[i,j] > scores[i,chosen_class]:
        chosen_class = j
    predictions[i] = chosen_class

  return predictions, np.count_nonzero(predictions == y.reshape(m)) * 100 / m

# Multi class clasifier using scikit
def scikit_train(train_data):
  x, y, m = get_input_all(train_data)
  classifier = svm.SVC(C=1, kernel='rbf', gamma=0.05)
  classifier.fit(x, y.reshape(m))
  return classifier

def scikit_test(classifier, test_data):
  x, y, m = get_input_all(test_data)
  predictions = classifier.predict(x)
  return predictions, 100 * metrics.accuracy_score(y.reshape(m), predictions)

def cross_validation(train_data, test_data):
  x, y, m = get_input_all(train_data)
  x_test, y_test, m_test = get_input_all(test_data)
  
  # dividing train data into 5 folds
  fold_size = m//5 # since we are doing 5-fold cross validation
  m_tr = m - fold_size # size of "training" set
  # cross validation sets
  x_cv = [x[i*fold_size : (i+1)*fold_size] for i in range(5)]
  y_cv = [y[i*fold_size : (i+1)*fold_size] for i in range(5)]
  # "training" sets (considering all blocks other than corresponding cv set)
  x_tr = [np.concatenate((x[ : i*fold_size], x[(i+1)*fold_size : ])) for i in range(5)]
  y_tr = [np.concatenate((y[ : i*fold_size], y[(i+1)*fold_size : ])) for i in range(5)]
  
  # C values to test
  C_vals = [10, 5, 1, 1e-3, 1e-5]

  # cv and test accuracies
  accuracy_cv = []
  accuracy_test = []

  for C_val in C_vals:
    acc_sum = 0
    for i in range(5):
      classifier = svm.SVC(C=C_val, kernel='rbf', gamma=0.05, decision_function_shape='ovo')
      classifier.fit(x_tr[i], y_tr[i].reshape(m_tr))
      # cv predictions
      cv_predictions = classifier.predict(x_cv[i])
      acc = 100 * metrics.accuracy_score(y_cv[i].reshape(fold_size), cv_predictions)
      acc_sum += acc
      print(C_val, i, acc)
    accuracy_cv.append(acc_sum/5)
    # test predictions
    classifier = svm.SVC(C=C_val, kernel='rbf', gamma=0.05, decision_function_shape='ovo')
    classifier.fit(x, y.reshape(m))
    test_predictions = classifier.predict(x_test)
    accuracy_test.append(100 * metrics.accuracy_score(y_test.reshape(m_test), test_predictions))
    print(C_val, accuracy_test[-1])

  return np.array(C_vals), np.array(accuracy_cv), np.array(accuracy_test)

def plot_cross_val(C_vals, accuracy_cv, accuracy_test):
  plt.figure()
  plt.title('Cross-validation and test accuracy vs C')
  plt.xlabel(r'log$_{10}$(C)')
  plt.ylabel('Accuracy (%)')
  plt.plot(np.log10(C_vals), accuracy_cv, label='5-fold cross validation')
  plt.plot(np.log10(C_vals), accuracy_test, label='Test set')
  plt.legend()
  plt.show()

def get_confusion_matrix(predictions, y, m):
    cm = np.zeros((10,10),dtype='int32')
    for i in range(m):
        cm[int(predictions[i]), int(y[i,0])] += 1
    return cm

# Data for binary classification
class1, class2 = 5, 6
x, y, m = get_input(train_data, class1, class2)
xv, yv, mv = get_input(val_data, class1, class2)
xt, yt, mt = get_input(test_data, class1, class2)

# Data for multiclass classification
xva, yva, mva = get_input_all(val_data)
xta, yta, mta = get_input_all(test_data)

# --- PART (a) ----
# --a.i. Linear SVM model
# Training
start_time = time()
alpha = linear_svm(x, y, m)
x_sup, y_sup, m_sup, alpha_sup = get_support_vectors(alpha, x, y, m)
print(m_sup, 'support vectors found')
w, b = get_params(x_sup, y_sup, m_sup, alpha_sup)
print('Training time:', time() - start_time)
# Testing on sets and reporting accuracy
start_time = time()
train_accuracy, train_predictions = get_accuracy_linear(w, b, m, x, y)
print('Train accuracy:', train_accuracy)
val_accuracy, val_predictions = get_accuracy_linear(w, b, mv, xv, yv)
print('Val accuracy:', val_accuracy)
test_accuracy, test_predictions = get_accuracy_linear(w, b, mt, xt, yt)
print('Test accuracy:', test_accuracy)
print('Prediction time:', time() - start_time)

# --a.ii. Gaussian kernel SVM model
# Training
start_time = time()
alpha_gauss = gaussian_kernel_svm(x, y, m)
x_sup, y_sup, m_sup, alpha_sup = get_support_vectors(alpha_gauss, x, y, m)
print(m_sup, 'support vectors found')
b_gauss = get_b_gaussian(x_sup, y_sup, m_sup, alpha_sup)
print('Training time:', time() - start_time)
# Testing
start_time = time()
train_accuracy, train_predictions = get_accuracy_gaussian(alpha_sup, b_gauss, m, x, y, x_sup, m_sup, y_sup)
print('Train accuracy:', train_accuracy)
val_accuracy, val_predictions = get_accuracy_gaussian(alpha_sup, b_gauss, mv, xv, yv, x_sup, m_sup, y_sup)
print('Val accuracy:', val_accuracy)
test_accuracy, test_predictions = get_accuracy_gaussian(alpha_sup, b_gauss, mt, xt, yt, x_sup, m_sup, y_sup)
print('Test accuracy:', test_accuracy)
print('Prediction time:', time() - start_time)

# --- Part (b) ---
# --b.i. Multi-class classifier using cvxopt gaussian
# Training
start_time = time()
alphas, bs = train_multi_classifier(train_data)
print('Training time:', time() - start_time)
# Testing
start_time = time()
val_predictions, val_accuracy = test_multi_class(val_data, train_data, alphas, bs)
print('Val accuracy:', val_accuracy)
test_predictions, test_accuracy = test_multi_class(test_data, train_data, alphas, bs)
print('Test accuracy:', test_accuracy)
print('Prediction time:', time() - start_time)
# --- Part (b.iii) ----
print('Confusion matrix (Val set):')
print(get_confusion_matrix(val_predictions, yva, mva))
print('Confusion matrix (Test set):')
print(get_confusion_matrix(test_predictions, yta, mta))

# Testing
start_time = time()
val_predictions, val_accuracy = test_multi_class(val_data, train_data, alphas, bs)
print('Val accuracy:', val_accuracy)
test_predictions, test_accuracy = test_multi_class(test_data, train_data, alphas, bs)
print('Test accuracy:', test_accuracy)
print('Prediction time:', time() - start_time)
# --- Part (b.iii) ----
print('Confusion matrix (Val set):')
print(get_confusion_matrix(val_predictions, yva, mva))
print('Confusion matrix (Test set):')
print(get_confusion_matrix(test_predictions, yta, mta))

# --b.ii. Scikit's multiclass SVM
# Training
start_time = time()
classifier = scikit_train(train_data)
print('Training time:', time() - start_time)
# Testing
start_time = time()
val_predictions, val_accuracy = scikit_test(classifier, val_data)
print('Val accuracy:', val_accuracy)
test_predictions, test_accuracy = scikit_test(classifier, test_data)
print('Test accuracy:', test_accuracy)
print('Prediction time:', time() - start_time)
# --- Part (b.iii) ----
print('Confusion matrix (Val set):')
print(get_confusion_matrix(val_predictions, yva, mva))
print('Confusion matrix (Test set):')
print(get_confusion_matrix(test_predictions, yta, mta))

# --b.iv. 5-fold cross validation
start_time = time()
C_vals, accuracy_cv, accuracy_test = cross_validation(train_data, test_data)
plot_cross_val(C_vals, accuracy_cv, accuracy_test)
print('Cross validation time:', time() - start_time)
