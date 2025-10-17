from classes import *
from funxs import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets
mnist = datasets.load_digits()
X,y = mnist.data, mnist.target.astype(int)


SAMPLES = 500
LEARNING_RATE = 0.02
EPOCHS = 100
MEAN1 = np.array([0, 0])
MEAN2 = np.array([5, 5])
MEAN3 = np.array([0, 5])
MEAN4 = np.array([5, 0])

COV = np.array([[0.5, 0], [0, 0.5]])

X1 = multivariate_normal.rvs(mean=MEAN1, cov=COV, size=SAMPLES)
X2 = multivariate_normal.rvs(mean=MEAN2, cov=COV, size=SAMPLES)
X3 = multivariate_normal.rvs(mean=MEAN3, cov=COV, size=SAMPLES)
X4 = multivariate_normal.rvs(mean=MEAN4, cov=COV, size=SAMPLES)

X_class0 = np.vstack((X1, X2))
y_class0 = np.zeros(len(X_class0))

X_class1 = np.vstack((X3, X4))
y_class1 = np.ones(len(X_class1))

X = np.vstack((X_class0, X_class1))
y = np.hstack((y_class0, y_class1))

indices = np.arange(X.shape[0])

np.random.shuffle(indices)

test_set_size = int(len(X) * 0.4)


test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]


X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]


n_features = X_train.shape[1]
n_output = 1

A = np.random.randn(n_output, n_features) * 0.1
b = np.zeros((n_output, 1))

x_node = Input()
y_node = Input()

A_node = Parameter(A)
b_node = Parameter(b)


linear_node = Linear(x_node, A_node, b_node)
sigmoid = Sigmoid(linear_node)
loss = BCE(y_node, sigmoid)

graph = [x_node, y_node, A_node, b_node, linear_node, sigmoid, loss]
trainable = [A_node, b_node]

x_node.value = X_train.T
y_node.value = y_train.reshape(1, -1)

for epoch in range(EPOCHS):
    forward_pass(graph)
    backward_pass(graph)
    sgd_update(trainable, LEARNING_RATE)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.value}")

x_node.value = X_test.T
y_node.value = y_test.reshape(1, -1)
forward_pass(graph)
predictions = (sigmoid.value > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.reshape(1, -1))

print(f"\nTest Loss: {loss.value}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

