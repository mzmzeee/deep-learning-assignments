from scipy.stats import multivariate_normal
import numpy as np
from classes import *
import matplotlib.pyplot as plt
import os

def forward_pass(graph, final_node=None):
    if final_node is None:
        final_node = graph[-1]
    
    for n in graph:
        n.forward()
        if n == final_node:
            break

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]
        
def gen_xordata(samples = 100 ,test_percent = 0.3, noise=0.1 ):
    MEAN1 = np.array([0, 0])
    MEAN2 = np.array([5, 5])
    MEAN3 = np.array([0, 5])
    MEAN4 = np.array([5, 0])

    COV = np.array([[noise, 0], [0, noise]])

    X1 = multivariate_normal.rvs(mean=MEAN1, cov=COV, size=samples)
    X2 = multivariate_normal.rvs(mean=MEAN2, cov=COV, size=samples)
    X3 = multivariate_normal.rvs(mean=MEAN3, cov=COV, size=samples)
    X4 = multivariate_normal.rvs(mean=MEAN4, cov=COV, size=samples)

    X_class0 = np.vstack((X1, X2))
    y_class0 = np.zeros(len(X_class0))

    X_class1 = np.vstack((X3, X4))
    y_class1 = np.ones(len(X_class1))

    X = np.vstack((X_class0, X_class1))
    y = np.hstack((y_class0, y_class1))

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(len(X) * test_percent)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def split_data(X, y, test_size=0.25, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def topological_sort(entry_node):
    visited = set()
    sorted_nodes = []

    def visit(node):
        if node not in visited:
            visited.add(node)
            for n in node.inputs:
                visit(n)
            sorted_nodes.append(node)

    visit(entry_node)
    return sorted_nodes

def get_trainable(graph):
    trainable_nodes = []
    for node in graph:
        if isinstance(node, Parameter):
            trainable_nodes.append(node)
    return trainable_nodes

def save_plot(filename):
    path = 'assingment1/figure'
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(os.path.join(path, filename))