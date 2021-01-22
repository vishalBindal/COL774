dir_path = './decision_tree/'

import numpy as np
import matplotlib.pyplot as plt
import csv
from math import inf
from sklearn.metrics import accuracy_score
import os
from sklearn.ensemble import RandomForestClassifier

n_nodes = 1

def get_data():
    train_data = np.genfromtxt(os.path.join(dir_path, 'train.csv'), delimiter=',')
    val_data = np.genfromtxt(os.path.join(dir_path, 'val.csv'), delimiter=',')
    test_data = np.genfromtxt(os.path.join(dir_path, 'test.csv'), delimiter=',')
    
    x_train, y_train = train_data[1:, :-1], train_data[1:,-1]
    x_val, y_val = val_data[1:, :-1], val_data[1:,-1]
    x_test, y_test = test_data[1:, :-1], test_data[1:,-1]
    
    cont_attr = [] # list of booleans denoting whether attribute is continuous
    with open(os.path.join(dir_path, 'train.csv')) as train_file:
        reader = csv.reader(train_file, delimiter=',')
        row1 = next(reader)
        for attr_label in row1[:-1]:
            if attr_label.endswith(':Continuous'):
                cont_attr.append(True)
            else:
                cont_attr.append(False)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, cont_attr

class Node:
    def __init__(self, class_label):
        self.id = None
        self.attr = None
        self.cont = None
        self.class_label = class_label
        self.children = []
    
    def addChild(self, node, attr_value):
        self.children.append((node, attr_value))

def attr_split(x, y, j, cont, get_x=True):
    x_l, y_l, x_r, y_r = [], [], [], [] # left and right children
    
    if not get_x:
      # if attribute j is discrete (boolean)
      if not cont:
        for i in range(len(y)):
            if x[i,j] == 1:
                y_l.append(y[i])
            else:
                y_r.append(y[i]) 
      # if attribute j is continuous
      else:
        median = np.median(x[:,j])
        for i in range(len(y)):
            if x[i,j] <= median:
                y_l.append(y[i])
            else:
                y_r.append(y[i]) 
      return [], y_l, [], y_r

    # if attribute j is discrete (boolean)
    if not cont:
        for i in range(len(y)):
            if x[i,j] == 1:
                x_l.append(x[i,:])
                y_l.append(y[i])
            else:
                x_r.append(x[i,:])
                y_r.append(y[i]) 

    # if attribute j is continuous
    else:
        median = np.median(x[:,j])
        for i in range(len(y)):
            if x[i,j] <= median:
                x_l.append(x[i,:])
                y_l.append(y[i])
            else:
                x_r.append(x[i,:])
                y_r.append(y[i]) 
    return np.array(x_l), np.array(y_l), np.array(x_r), np.array(y_r)

def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y) # Probability of each class
    return -1*np.sum(probs * np.log(probs))

def chooseBestAttrToSplit(x, y, cont_attr):
    k = x.shape[1] # no of attributes
    min_entropy, j_best = inf, None
    for j in range(k):
        _, y_l, _, y_r = attr_split(x, y, j, cont_attr[j], get_x = False)
        if len(y_l)>0 and len(y_r)>0:
            entropy_j = calculate_entropy(y_l) + calculate_entropy(y_r)
            if entropy_j < min_entropy:
                min_entropy = entropy_j
                j_best = j
    return j_best

def most_occuring(arr):
    vals, counts = np.unique(arr, return_counts=True)
    ind = np.argmax(counts)
    return vals[ind]
    
def grow_tree(root, x, y, cont_attr):
    print('Growing tree for', len(y), 'examples')
    global n_nodes
    root.id = n_nodes
    n_nodes += 1
    
    # If all examples belong to the same class
    if np.all(y == y[0]):
        root.class_label = y[0]
        return
    
    j = chooseBestAttrToSplit(x, y, cont_attr)
    if j is None:
        root.class_label = most_occuring(y)
        return
    print('Splitting on attr', j)
    root.attr = j
    root.cont = cont_attr[j]
    x_l, y_l, x_r, y_r = attr_split(x, y, j, cont_attr[j])
    
    leftChild = Node(most_occuring(y_l))
    rightChild = Node(most_occuring(y_r))
    print('Left child:', len(y_l))
    print('Right child:', len(y_r))
    # if attribute j is discrete (boolean)
    if not cont_attr[j]:
        root.addChild(leftChild, 1)
        root.addChild(rightChild, 0)

    # if attribute j is continuous
    else:
        median = np.max(x_l[:,j])
        root.addChild(leftChild, median)
        root.addChild(rightChild, None)

    # further grow the tree
    grow_tree(leftChild, x_l, y_l, cont_attr)
    grow_tree(rightChild, x_r, y_r, cont_attr)

def decision_tree_predict(root, x_example):
    j = root.attr
    if j is None:
        return root.class_label
    # continuous attr split
    if root.cont:
        # if example's attr value <= median
        if x_example[j] <= root.children[0][1]:
            return decision_tree_predict(root.children[0][0], x_example)
        return decision_tree_predict(root.children[1][0], x_example)
    # discrete attr split
    # if example's attr value == 1
    if x_example[j] == root.children[0][1]:
        return decision_tree_predict(root.children[0][0], x_example)
    return decision_tree_predict(root.children[1][0], x_example)
    
def predict(decision_tree, x_test):
    predictions = np.zeros(x_test.shape[0], dtype='int8')
    for i in range(x_test.shape[0]):
        predictions[i] = decision_tree_predict(decision_tree, x_test[i,:])
    return predictions
        
def get_accuracies(decision_tree, n_nodes, x_train, y_train, x_val, y_val, x_test, y_test):
    print(n_nodes)
    predictions = predict(decision_tree, x_train)
    print('Train set accuracy:', 100*accuracy_score(y_train, predictions))
    predictions = predict(decision_tree, x_val)
    print('Val set accuracy:', 100*accuracy_score(y_val, predictions))
    predictions = predict(decision_tree, x_test)
    print('Test set accuracy:', 100*accuracy_score(y_test, predictions))
  
def get_grow_accuracies(x, y, decision_tree, n_nodes):
    correct_predictions = np.zeros(n_nodes, dtype='int32')
    majority_class = most_occuring(y)
    correct_predictions[0] = np.count_nonzero(y == majority_class)
    for i in range(len(y)):
        if i%10000 == 0:
            print(i)
        example, label = x[i,:], y[i]
        root = decision_tree
        while root.attr is not None:
            if root.cont:
                if example[root.attr] <= root.children[0][1]:
                    new_root = root.children[0][0]
                else:
                    new_root = root.children[1][0]
            else:
                if example[root.attr] == root.children[0][1]:
                    new_root = root.children[0][0]
                else:
                    new_root = root.children[1][0]
            correct_predictions[root.id:new_root.id] += (new_root.class_label == label)
            root = new_root
        correct_predictions[root.id:] += (root.class_label == label)
    return correct_predictions

def plot_grow_accuracies(train_accuracies, val_accuracies, test_accuracies):
    plt.figure()
    plt.xlabel('No of nodes in decision tree')
    plt.ylabel('Accuracy (%)')
    count_nodes = np.array([i for i in range(len(train_accuracies))])
    plt.plot(count_nodes, train_accuracies, label='Train set')
    plt.plot(count_nodes, val_accuracies, label='Validation set', color='black')
    plt.plot(count_nodes, test_accuracies, label='Test set')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'tree_grow_accuracies.png'), dpi=3000)
    plt.show()

def get_subtree_size(root):
    if len(root.children) == 0:
        return 0
    return 2 + get_subtree_size(root.children[0][0]) + get_subtree_size(root.children[1][0])

def get_nodewise_classification(decision_tree, x, y, correct_class_examples):
    majority_class = most_occuring(y)
    correct_class_examples[0] = np.count_nonzero(y == majority_class)
    for i in range(len(y)):
        example, label = x[i,:], y[i]
        root = decision_tree
        while root.attr is not None:
            if root.cont:
                if example[root.attr] <= root.children[0][1]:
                    new_root = root.children[0][0]
                else:
                    new_root = root.children[1][0]
            else:
                if example[root.attr] == root.children[0][1]:
                    new_root = root.children[0][0]
                else:
                    new_root = root.children[1][0]
            root = new_root
            correct_class_examples[root.id] += (root.class_label == label)

def prune_tree(root, c_val, c_train, c_test, prune_points):
    if root.attr is None:
        # if leaf node, return
        return
    leftChild, rightChild = root.children[0][0], root.children[1][0]
    # prune left subtree
    prune_tree(leftChild, c_val, c_train, c_test, prune_points)
    # prune right subtree
    prune_tree(rightChild, c_val, c_train, c_test, prune_points)
    # if correct classifications in left and right subtrees are less, then we can prune
    # Here c_val[child] (LHS) corresponds to whole subtree of child,
    # whereas c_val[root] (RHS) corresponds to only classification at root
    if c_val[leftChild.id] + c_val[rightChild.id] <= c_val[root.id]:
        # previous accuracies
        n_nodes, acc_v, acc_tr, acc_te = prune_points[-1]
        # calculate new accuracies (correctly classified examples)
        # nodes_pruned = get_subtree_size(leftChild) + get_subtree_size(rightChild)
        nodes_pruned = get_subtree_size(root)
        n_nodes -= nodes_pruned
        acc_v += c_val[root.id] - c_val[leftChild.id] - c_val[rightChild.id]
        acc_tr += c_train[root.id] - c_train[leftChild.id] - c_train[rightChild.id]
        acc_te += c_test[root.id] - c_test[leftChild.id] - c_test[rightChild.id]
        prune_points.append((n_nodes, acc_v, acc_tr, acc_te))
        # prune
        root.attr = None
        root.children = []   
    else:
        # no pruning, just update c_val, c_train and c_test to reflect whole subtree
        c_val[root.id] = c_val[leftChild.id] + c_val[rightChild.id]
        c_train[root.id] = c_train[leftChild.id] + c_train[rightChild.id]
        c_test[root.id] = c_test[leftChild.id] + c_test[rightChild.id]    


def plot_prune_accuracies(train_accuracies, val_accuracies, test_accuracies, prune_nodes, prune_val_acc, prune_tr_acc, prune_te_acc):
    plt.xlabel('No of nodes in decision tree')
    plt.ylabel('Accuracy (%)')
    count_nodes = np.array([i for i in range(len(train_accuracies))])
    plt.plot(count_nodes, train_accuracies, label='Train set', color='blue')
    plt.plot(count_nodes, val_accuracies, label='Validation set', color='black')
    plt.plot(count_nodes, test_accuracies, label='Test set', color='orange')

    plt.plot(prune_nodes, prune_tr_acc, label='Train set (pruning)', color='blue', linestyle='dashed')
    plt.plot(prune_nodes, prune_val_acc, label='Validation set (pruning)', color='black', linestyle='dashed')
    plt.plot(prune_nodes, prune_te_acc, label='Test set (pruning)', color='orange', linestyle='dashed')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'tree_prune_accuracies.png'), dpi=3000)


def grid_search_tuning(x_train, y_train):
    parameters = {
        'n_estimators': [50, 150, 250, 350, 450],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_samples_split': [2, 4, 6, 8, 10]
    }

    best_score = -inf
    best_n_estimators, best_max_features, best_min_samples_split = None, None, None
    for n_estimators in parameters['n_estimators']:
        for max_features in parameters['max_features']:
            for min_samples_split in parameters['min_samples_split']:
                rfc = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split, criterion='entropy', oob_score=True)
                rfc.fit(x_train, y_train)
                print(n_estimators, max_features, min_samples_split, rfc.oob_score_)
                if rfc.oob_score_ > best_score:
                    print('This is good!')
                    best_n_estimators = n_estimators
                    best_max_features = max_features
                    best_min_samples_split = min_samples_split
                    best_score = rfc.oob_score_
    return best_n_estimators, best_max_features, best_min_samples_split, best_score
  
def best_rf_model(x_train, y_train, x_val, y_val, x_test, y_test):
    rfc = RandomForestClassifier(n_estimators=350, max_features=0.7, min_samples_split=2, criterion='entropy', oob_score=True)
    rfc.fit(x_train, y_train)
    print('Out-of-bag accuracy:', 100*rfc.oob_score_)
    predictions = rfc.predict(x_train)
    print('Train accuracy:', 100*accuracy_score(y_train, predictions))
    predictions = rfc.predict(x_val)
    print('Val accuracy:', 100*accuracy_score(y_val, predictions))
    predictions = rfc.predict(x_test)
    print('Test accuracy:', 100*accuracy_score(y_test, predictions))

def get_rf_accuracies(rfc, x_train, y_train, x_val, y_val, x_test, y_test):
    rfc.fit(x_train, y_train)
    predictions = rfc.predict(x_val)
    val_accuracy = 100*accuracy_score(y_val, predictions)
    predictions = rfc.predict(x_test)
    test_accuracy = 100*accuracy_score(y_test, predictions)
    return val_accuracy, test_accuracy

def plot_sensitivity_analysis(param_array, val_accuracies, test_accuracies, param_name):
    fig, ax = plt.subplots()
    ax.plot(param_array, np.array(val_accuracies), label='Val set accuracy')
    ax.plot(param_array, np.array(test_accuracies), label='Test set accuracy')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Parameter sensitivity analysis')
    ax.legend()
    fig.savefig(f'{param_name}.png', dpi=3000)
    

def sensitivity_analysis(x_train, y_train, x_val, y_val, x_test, y_test):
    # n_estimators
    params = np.array([50, 150, 250, 350, 450])
    val_accuracies, test_accuracies = [], []
    for param in params:
        rfc = RandomForestClassifier(n_estimators=param, max_features=0.7, min_samples_split=2, criterion='entropy')
        val_accuracy, test_accuracy = get_rf_accuracies(rfc, x_train, y_train, x_val, y_val, x_test, y_test)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        print(param, val_accuracy, test_accuracy)
    plot_sensitivity_analysis(params, val_accuracies, test_accuracies, 'n_estimators')

    # max_features
    params = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    val_accuracies, test_accuracies = [], []
    for param in params:
        rfc = RandomForestClassifier(n_estimators=350, max_features=param, min_samples_split=2, criterion='entropy')
        val_accuracy, test_accuracy = get_rf_accuracies(rfc, x_train, y_train, x_val, y_val, x_test, y_test)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        print(param, val_accuracy, test_accuracy)
    plot_sensitivity_analysis(params, val_accuracies, test_accuracies, 'max_features')

    # min_samples_split
    params = np.array([2, 4, 6, 8, 10])
    val_accuracies, test_accuracies = [], []
    for param in params:
        rfc = RandomForestClassifier(n_estimators=350, max_features=0.7, min_samples_split=param, criterion='entropy')
        val_accuracy, test_accuracy = get_rf_accuracies(rfc, x_train, y_train, x_val, y_val, x_test, y_test)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        print(param, val_accuracy, test_accuracy)
    plot_sensitivity_analysis(params, val_accuracies, test_accuracies, 'min_samples_split')


x_train, y_train, x_val, y_val, x_test, y_test, cont_attr = get_data()

decision_tree = Node(most_occuring(y_train))
grow_tree(decision_tree, x_train, y_train, cont_attr)

get_accuracies(decision_tree, n_nodes, x_train, y_train, x_val, y_val, x_test, y_test)

train_corr_preds = get_grow_accuracies(x_train, y_train, decision_tree, n_nodes)
train_accuracies = 100*train_corr_preds/len(y_train)
val_corr_preds = get_grow_accuracies(x_val, y_val, decision_tree, n_nodes)
val_accuracies = 100*val_corr_preds/len(y_val)
test_corr_preds = get_grow_accuracies(x_test, y_test, decision_tree, n_nodes)
test_accuracies = 100*test_corr_preds/len(y_test)

plot_grow_accuracies(train_accuracies, val_accuracies, test_accuracies)

c_val = np.zeros(n_nodes, dtype='int32')
c_train = np.zeros(n_nodes, dtype='int32')
c_test = np.zeros(n_nodes, dtype='int32')
get_nodewise_classification(decision_tree, x_val, y_val, c_val)
get_nodewise_classification(decision_tree, x_train, y_train, c_train)
get_nodewise_classification(decision_tree, x_test, y_test, c_test)
prune_points = [(n_nodes, val_corr_preds[-1], train_corr_preds[-1], test_corr_preds[-1])]
prune_tree(decision_tree, c_val, c_train, c_test, prune_points)
print(prune_points)
prune_nodes, prune_val_preds, prune_train_preds, prune_test_preds = np.array(prune_points).T
nodes_remaining = prune_nodes[-1]
print('Final nodes left after pruning:', nodes_remaining)
prune_val_accs, prune_train_accs, prune_test_accs = 100*prune_val_preds/len(y_val), 100*prune_train_preds/len(y_train), 100*prune_test_preds/len(y_test)

plot_prune_accuracies(train_accuracies, val_accuracies, test_accuracies, prune_nodes, prune_val_accs, prune_train_accs, prune_test_accs)

get_accuracies(decision_tree, nodes_remaining, x_train, y_train, x_val, y_val, x_test, y_test)

print('Starting grid search..')
best_n_estimators, best_max_features, best_min_samples_split, best_score = grid_search_tuning(x_train, y_train)
print('Optimal set of parameters:')
print('n_estimators:', best_n_estimators)
print('max_features:', best_max_features)
print('min_samples_split:', best_min_samples_split)
print('oob_score', best_score)

best_rf_model(x_train, y_train, x_val, y_val, x_test, y_test)

sensitivity_analysis(x_train, y_train, x_val, y_val, x_test, y_test)