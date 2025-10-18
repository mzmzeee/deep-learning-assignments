from classes import *
from funxs import *
import numpy as np

SAMPLES = 500
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_PERCENT =0.4

X_train, X_test ,y_train, y_test = gen_xordata(SAMPLES , TEST_PERCENT)
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

