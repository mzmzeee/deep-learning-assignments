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

probabilities = output.value
metrics = compute_classification_metrics(y_test, predictions, num_classes=10)
print("Classification summary:")
print(format_classification_metrics(metrics, class_names=mnist.target_names))

rng = np.random.default_rng(13)
sample_count = min(8, len(y_test))
sample_indices = rng.choice(len(y_test), size=sample_count, replace=False)
print("Sample deep CNN predictions with confidence:")
for idx in sample_indices:
    confidence = probabilities[predictions[idx], idx]
    print(f" idx {idx:4d} | true={y_test[idx]} pred={predictions[idx]} conf={confidence:.3f}")

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(loss_history)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Best CNN Training Loss')
ax_loss.grid(True)
save_plot('c_task4_best_cnn_loss.png')
plt.close(fig_loss)

cm = metrics['confusion_matrix']

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

plot_sample_predictions(X_test[:, 0, :, :], y_test, predictions, 'c_task4_sample_predictions.png', num_samples=12, class_names=mnist.target_names, random_state=9)

log_metrics('c_task4', {
    'accuracy': float(metrics['accuracy']),
    'macro_f1': float(metrics['macro_f1']),
    'min_training_loss': float(np.min(loss_history)),
    'final_training_loss': float(loss_history[-1])
})

plot_metric_summary(metric_key='accuracy', filename='c_tasks_accuracy.png', title='C Tasks Accuracy Comparison')
plot_metric_summary(metric_key='macro_f1', filename='c_tasks_macro_f1.png', title='C Tasks Macro F1 Comparison')
best_task, best_value = get_best_metric('accuracy')
if best_task is not None:
    print(f"Highest recorded accuracy so far: {best_task} with {best_value:.2%}")
