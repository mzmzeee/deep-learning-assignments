from classes import *
from funxs import *
import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 400
LEARNING_RATE = 0.7
EPOCHS = 1000
TEST_PERCENT = 0.4

X_train, X_test, y_train, y_test = gen_xordata(SAMPLES, TEST_PERCENT, noise=1)
n_features = X_train.shape[1]
n_hidden_1 = 20
n_hidden_2 = 20
n_output = 1

x_node = Input()
y_node = Input()

linear1 = AutomatedLinear(x_node, n_features, n_hidden_1)
sigmoid1 = Sigmoid(linear1)
linear2 = AutomatedLinear(sigmoid1, n_hidden_1, n_hidden_2)
sigmoid2 = Sigmoid(linear2)
linear3 = AutomatedLinear(sigmoid2, n_hidden_2, n_output)
output_sigmoid = Sigmoid(linear3)
loss = BCE(y_node, output_sigmoid)

graph = topological_sort(loss)
trainable = get_trainable(graph)

x_node.value = X_train.T
y_node.value = y_train.reshape(1, -1)

losses = []
for epoch in range(EPOCHS):
    forward_pass(graph)
    backward_pass(graph)
    sgd_update(trainable, LEARNING_RATE)
    
    loss_value = loss.value
    losses.append(loss_value)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss_value}")

x_node.value = X_test.T
y_node.value = y_test.reshape(1, -1)
forward_pass(graph)
predictions = (output_sigmoid.value > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.reshape(1, -1))

print(f"\nTest Loss: {loss.value}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral, s=20)
plt.title("Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(2, 2, 2)
x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_data = np.c_[xx.ravel(), yy.ravel()]

x_node.value = grid_data.T
forward_pass(graph, final_node=output_sigmoid)
Z = (output_sigmoid.value > 0.5).astype(int)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral, s=20)
plt.title("Decision Boundary on Test Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(2, 2, 3)
plt.plot(range(EPOCHS), losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

ax = plt.subplot(2, 2, 4)
y_true = y_test.reshape(1, -1)
y_pred = predictions

tp = np.sum((y_true == 1) & (y_pred == 1))
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))

cm = np.array([[tn, fp], [fn, tp]])

im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'],
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
