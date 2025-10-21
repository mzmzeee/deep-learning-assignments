from classes import *
from funxs import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)

CLASS1_SIZE = 140
CLASS2_SIZE = 140

learning_rate = 0.5 
epochs = 100 
TEST_SIZE = 0.25
BATCH_SIZE = [1, 4, 8, 16, 64, 128]

COV_SCALE = 2.1
MEAN1 = np.array([0.6, 1.0])
COV1 = COV_SCALE * np.array([[1.0, 0.7],
                             [0.7, 1.0]])
MEAN2 = np.array([-0.4, -1.0])
COV2 = COV_SCALE * np.array([[1.0, -0.7],
                             [-0.7, 1.0]])

X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))


X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE)

n_features = X_train.shape[1]
n_output = 1

batch_losses = []
batch_accuracies = {}


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

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

axs[0, 0].scatter(X[y==0][:, 0], X[y==0][:, 1], label='Class 0')
axs[0, 0].scatter(X[y==1][:, 0], X[y==1][:, 1], label='Class 1')
axs[0, 0].set_title('Data Distribution')
axs[0, 0].set_xlabel('Feature 1')
axs[0, 0].set_ylabel('Feature 2')
axs[0, 0].legend()
axs[0, 0].grid(True)


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
        if (epoch + 1) % 10 == 0: 
            print(f"Epoch {epoch + 1}, Loss: {final_loss}")
    
    axs[1, 1].plot(range(epochs), epoch_losses, label=f'batch={current_batch_size}')

    correct_predictions = 0
    x_node.value = X_test.T
    forward_pass(graph, sigmoid)
    predictions = np.round(sigmoid.value)
    correct_predictions = np.sum(predictions == y_test.reshape(1, -1))

    accuracy = correct_predictions / X_test.shape[0]
    print(f"Accuracy for batch size {current_batch_size}: {accuracy * 100:.2f}%")
    batch_accuracies[current_batch_size] = accuracy
    # Build decision boundary and per-batch visualizations
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_node.value = grid.T
    forward_pass(graph, sigmoid)
    Z = sigmoid.value.reshape(xx.shape)

    # Confusion matrix from test predictions
    y_test_flat = y_test.flatten()
    predictions_flat = predictions.flatten()
    tn = np.sum((y_test_flat == 0) & (predictions_flat == 0))
    fp = np.sum((y_test_flat == 0) & (predictions_flat == 1))
    fn = np.sum((y_test_flat == 1) & (predictions_flat == 0))
    tp = np.sum((y_test_flat == 1) & (predictions_flat == 1))
    cm = np.array([[tn, fp], [fn, tp]])

    # Show in combined figure only for the last batch (to avoid clutter)
    if current_batch_size == BATCH_SIZE[-1]:
        axs[0, 1].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        axs[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolors='k')
        axs[0, 1].set_title(f'Decision Boundary (batch={current_batch_size})')
        axs[0, 1].set_xlabel('Feature 1')
        axs[0, 1].set_ylabel('Feature 2')

        im = axs[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[1, 0].set_title(f'Confusion Matrix (batch={current_batch_size})')
        tick_marks = np.arange(2)
        axs[1, 0].set_xticks(tick_marks, ['Class 0', 'Class 1'])
        axs[1, 0].set_yticks(tick_marks, ['Class 0', 'Class 1'])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[1, 0].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
        axs[1, 0].set_ylabel('True label')
        axs[1, 0].set_xlabel('Predicted label')

    # Save per-batch decision boundary
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolors='k')
    plt.title(f'Decision Boundary (batch={current_batch_size})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    save_plot(f"a_decision_boundary_batch_{current_batch_size}.png")
    plt.close()

    # Save per-batch confusion matrix
    fig_cm, ax_cm = plt.subplots()
    im_cm = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title(f'Confusion Matrix (batch={current_batch_size})')
    fig_cm.colorbar(im_cm, ax=ax_cm)
    tick_marks = np.arange(2)
    ax_cm.set_xticks(tick_marks, ['Class 0', 'Class 1'])
    ax_cm.set_yticks(tick_marks, ['Class 0', 'Class 1'])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    ax_cm.set_ylabel('True label')
    ax_cm.set_xlabel('Predicted label')
    save_plot(f"a_confusion_matrix_batch_{current_batch_size}.png")
    plt.close()


accuracy_title = "Accuracies: " + ", ".join([f"batch {b}: {a*100:.2f}%" for b, a in batch_accuracies.items()])
fig.suptitle(accuracy_title)

axs[1, 1].set_title('Loss vs Epoch for Different Batch Sizes')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Training Loss')
axs[1, 1].legend()
axs[1, 1].grid(True)

fig.tight_layout(pad=3.0)
save_plot('assignment1_a_combined.png')
plt.show()
