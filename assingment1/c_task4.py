import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from funxs import *
from classes import *

mnist = load_digits()
X = mnist.data.astype(np.float32) / 16.0
y = mnist.target

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, seed=42)
X_train = X_train.reshape(-1, 1, 8, 8)
X_test = X_test.reshape(-1, 1, 8, 8)
y_train_one_hot = one_hot(y_train, 10)
y_test_one_hot = one_hot(y_test, 10)

x = Input()
conv1 = Conv(x, num_filters=16, input_channels=1, filter_size=3, stride=1, padding=1)
act1 = ReLU(conv1)
pool1 = MaxPooling(act1, pool_size=2, stride=2)

conv2 = Conv(pool1, num_filters=32, input_channels=16, filter_size=3, stride=1, padding=1)
act2 = ReLU(conv2)
pool2 = MaxPooling(act2, pool_size=2, stride=2)

conv3 = Conv(pool2, num_filters=64, input_channels=32, filter_size=3, stride=1, padding=1)
act3 = ReLU(conv3)
pool3 = MaxPooling(act3, pool_size=2, stride=2)

flatten = Flatten(pool3)

n_flat = 64 * 1 * 1
weights = np.random.randn(10, n_flat) * np.sqrt(1 / n_flat)
A = Parameter(weights)
b = Parameter(np.zeros((10, 1)))
linear = Linear(flatten, A, b)
output = Softmax(linear)

y_true = Input()
cost = CE(y_true, output)

graph = topological_sort(cost)
trainables = get_trainable(graph)

epochs = 80
batch_size = 32
learning_rate = 0.02
loss_history = []

num_samples = X_train.shape[0]

for _ in range(epochs):
    indices = np.random.permutation(num_samples)
    epoch_loss = 0.0
    batches = 0
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        X_batch = X_train[batch_idx]
        y_batch = y_train_one_hot[:, batch_idx]

        x.forward(X_batch)
        y_true.forward(y_batch)
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, learning_rate)

        epoch_loss += cost.value
        batches += 1

    loss_history.append(epoch_loss / batches)

x.forward(X_test)
y_true.forward(y_test_one_hot)
forward_pass(graph, final_node=output)
predictions = np.argmax(output.value, axis=0)
accuracy = np.mean(predictions == y_test)
print(f"Best CNN Test Accuracy: {accuracy:.4f}")

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(loss_history)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Best CNN Training Loss')
ax_loss.grid(True)
save_plot('c_task4_best_cnn_loss.png')
plt.close(fig_loss)

cm = np.zeros((10, 10), dtype=int)
for true_label, pred_label in zip(y_test, predictions):
    cm[true_label, pred_label] += 1

fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, cmap=plt.cm.Blues)
fig_cm.colorbar(im, ax=ax_cm)
ax_cm.set_xticks(range(10))
ax_cm.set_yticks(range(10))
ax_cm.set_xlabel('Predicted label')
ax_cm.set_ylabel('True label')
ax_cm.set_title('Best CNN Confusion Matrix')
for i in range(10):
    for j in range(10):
        ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
save_plot('c_task4_best_cnn_cm.png')
plt.close(fig_cm)
