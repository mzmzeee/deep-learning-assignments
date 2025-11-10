import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from funxs import *
from classes import *

mnist = load_digits()
X = mnist.data.astype(np.float32) / 16.0
y = mnist.target

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, seed=42)
y_train_one_hot = one_hot(y_train, 10)
y_test_one_hot = one_hot(y_test, 10)

x = Input()
n_input = X_train.shape[1]
n_hidden = 128
n_output = 10

w1 = np.random.randn(n_hidden, n_input) * np.sqrt(1 / n_input)
w2 = np.random.randn(n_output, n_hidden) * np.sqrt(1 / n_hidden)

A1 = Parameter(w1)
b1 = Parameter(np.zeros((n_hidden, 1)))
A2 = Parameter(w2)
b2 = Parameter(np.zeros((n_output, 1)))
trainables = [A1, b1, A2, b2]

l1 = Linear(x, A1, b1)
a1 = ReLU(l1)
l2 = Linear(a1, A2, b2)
output = Softmax(l2)

y_true = Input()
cost = CE(y_true, output)

graph = topological_sort(cost)
losses = []
epochs = 1000
learning_rate = 0.05

for _ in range(epochs):
    x.forward(X_train.T)
    y_true.forward(y_train_one_hot)
    forward_pass(graph)
    backward_pass(graph)
    sgd_update(trainables, learning_rate)
    losses.append(cost.value)

x.forward(X_test.T)
y_true.forward(y_test_one_hot)
forward_pass(graph, final_node=output)
predictions = np.argmax(output.value, axis=0)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

probabilities = output.value
metrics = compute_classification_metrics(y_test, predictions, num_classes=10)
print("Classification summary:")
print(format_classification_metrics(metrics, class_names=mnist.target_names))

rng = np.random.default_rng(7)
sample_count = min(8, len(y_test))
sample_indices = rng.choice(len(y_test), size=sample_count, replace=False)
print("Sample predictions with confidence:")
for idx in sample_indices:
    confidence = probabilities[predictions[idx], idx]
    print(f" idx {idx:4d} | true={y_test[idx]} pred={predictions[idx]} conf={confidence:.3f}")

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(losses)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Training Loss')
ax_loss.grid(True)
save_plot('c_task1_loss.png')
plt.close(fig_loss)

cm = metrics['confusion_matrix']

fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, cmap=plt.cm.Blues)
fig_cm.colorbar(im, ax=ax_cm)
ax_cm.set_xticks(range(10))
ax_cm.set_yticks(range(10))
ax_cm.set_xlabel('Predicted label')
ax_cm.set_ylabel('True label')
ax_cm.set_title('Confusion Matrix')
for i in range(10):
    for j in range(10):
        ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
save_plot('c_task1_confusion_matrix.png')
plt.close(fig_cm)

test_images = X_test.reshape(-1, 8, 8)
plot_sample_predictions(test_images, y_test, predictions, 'c_task1_sample_predictions.png', num_samples=12, class_names=mnist.target_names, random_state=3)

log_metrics('c_task1', {
    'accuracy': float(metrics['accuracy']),
    'macro_f1': float(metrics['macro_f1']),
    'min_training_loss': float(np.min(losses)),
    'final_training_loss': float(losses[-1])
})

plot_metric_summary(metric_key='accuracy', filename='c_tasks_accuracy.png', title='C Tasks Accuracy Comparison')
plot_metric_summary(metric_key='macro_f1', filename='c_tasks_macro_f1.png', title='C Tasks Macro F1 Comparison')
best_task, best_value = get_best_metric('accuracy')
if best_task is not None:
    print(f"Current best recorded accuracy: {best_task} with {best_value:.2%}")
