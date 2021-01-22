import numpy as np
import csv
import sys
from math import inf
import os

n_nodes = 1

def get_data(train_path, val_path, test_path):
    train_data = np.genfromtxt(train_path, delimiter=',')
    val_data = np.genfromtxt(val_path, delimiter=',')
    test_data = np.genfromtxt(test_path, delimiter=',')
    
    x_train, y_train = train_data[1:, :-1], train_data[1:,-1]
    x_val, y_val = val_data[1:, :-1], val_data[1:,-1]
    x_test, y_test = test_data[1:, :-1], test_data[1:,-1]
    
    cont_attr = [] # list of booleans denoting whether attribute is continuous
    with open(train_path) as train_file:
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
    root.attr = j
    root.cont = cont_attr[j]
    x_l, y_l, x_r, y_r = attr_split(x, y, j, cont_attr[j])
    
    leftChild = Node(most_occuring(y_l))
    rightChild = Node(most_occuring(y_r))
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

def prune_tree(root, c_val):
    if root.attr is None:
        # if leaf node, return
        return
    leftChild, rightChild = root.children[0][0], root.children[1][0]
    # prune left subtree
    prune_tree(leftChild, c_val)
    # prune right subtree
    prune_tree(rightChild, c_val)
    # if correct classifications in left and right subtrees are less, then we can prune
    # Here c_val[child] (LHS) corresponds to whole subtree of child,
    # whereas c_val[root] (RHS) corresponds to only classification at root
    if c_val[leftChild.id] + c_val[rightChild.id] <= c_val[root.id]:
        # prune
        root.attr = None
        root.children = []   
    else:
        # no pruning, just update c_val, c_train and c_test to reflect whole subtree
        c_val[root.id] = c_val[leftChild.id] + c_val[rightChild.id]  


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


def run(question, train_data, val_data, test_data):
    x_train, y_train, x_val, y_val, x_test, y_test, cont_attr = get_data(train_data, val_data, test_data)
    decision_tree = Node(most_occuring(y_train))
    grow_tree(decision_tree, x_train, y_train, cont_attr)
    if int(question) == 2:
        c_val = np.zeros(n_nodes, dtype='int32')
        get_nodewise_classification(decision_tree, x_val, y_val, c_val)
        prune_tree(decision_tree, c_val)
    predictions = predict(decision_tree, x_test)
    return predictions


def main():
    question = sys.argv[1]
    train_data = sys.argv[2]
    val_data = sys.argv[3]
    test_data = sys.argv[4]
    output_file = sys.argv[5]
    output = run(question, train_data, val_data, test_data)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()
