from assingment1.classes import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

CLASS1_SIZE = 1000000
CLASS2_SIZE = 1000000
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25
BATCH_SIZE = 100000

MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
#plt.show()

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
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

epochs = EPOCHS
learning_rate = LEARNING_RATE

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


for epoch in range(epochs):
    loss_value = 0
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    for i in range(0, X_train.shape[0], BATCH_SIZE):
        X_batch = X_train_shuffled[i:i+BATCH_SIZE]
        y_batch = y_train_shuffled[i:i+BATCH_SIZE]

        x_node.value = X_batch.T
        y_node.value = y_batch.reshape(1, -1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value * X_batch.shape[0]

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

correct_predictions = 0
x_node.value = X_test.T
forward_pass(graph, sigmoid)
predictions = np.round(sigmoid.value)
correct_predictions = np.sum(predictions == y_test.reshape(1, -1))

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
grid = np.c_[xx.ravel(), yy.ravel()]
x_node.value = grid.T
forward_pass(graph, sigmoid)
Z = sigmoid.value.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
#plt.show()
