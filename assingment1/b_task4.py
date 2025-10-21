from sklearn import datasets
from funxs import *
from classes import * 
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.load_digits()
X, y = mnist.data , mnist.target

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4)
