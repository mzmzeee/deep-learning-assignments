from classes import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

CLASS1_SIZE = 1000
CLASS2_SIZE = 1000
N_FEATURES = 2
N_OUTPUT = 1
learning_rate = 0.02
epochs = 100
TEST_SIZE = 0.25
BATCH_SIZE = [1,2,4,8,16,32,64,128]

MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

n_features = X_train.shape[1]
n_output = 1

batch_losses = []


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

plt.figure()

for current_batch_size in BATCH_SIZE:
    print(f"Training with batch size: {current_batch_size}")
    A_node.value = np.random.randn(n_output, n_features) * 0.1
    b_node.value = np.zeros((n_output, 1))
    
    epoch_losses = []
    for epoch in range(epochs):
        loss_value = 0
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, X_train.shape[0], current_batch_size):
            X_batch = X_train_shuffled[i:i+current_batch_size]
            y_batch = y_train_shuffled[i:i+current_batch_size]

            x_node.value = X_batch.T
            y_node.value = y_batch.reshape(1, -1)

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, learning_rate)

            loss_value += loss.value * X_batch.shape[0]

        final_loss = loss_value / X_train.shape[0]
        epoch_losses.append(final_loss)
        if (epoch + 1) % 10 == 0: # Print loss every 10 epochs to reduce verbosity
            print(f"Epoch {epoch + 1}, Loss: {final_loss}")
    
    plt.plot(range(epochs), epoch_losses, label=f'batch={current_batch_size}')

    correct_predictions = 0
    x_node.value = X_test.T
    forward_pass(graph, sigmoid)
    predictions = np.round(sigmoid.value)
    correct_predictions = np.sum(predictions == y_test.reshape(1, -1))

    accuracy = correct_predictions / X_test.shape[0]
    print(f"Accuracy for batch size {current_batch_size}: {accuracy * 100:.2f}%")

    if current_batch_size == BATCH_SIZE[-1]:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
        grid = np.c_[xx.ravel(), yy.ravel()]
        x_node.value = grid.T
        forward_pass(graph, sigmoid)
        Z = sigmoid.value.reshape(xx.shape)

plt.title('Loss vs Epoch for Different Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_vs_batchsize.png')
plt.show()