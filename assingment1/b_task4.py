from sklearn import datasets
from funxs import *
from classes import * 
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.load_digits()
X, y = mnist.data , mnist.target

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4, seed=42)

y_train_one_hot = one_hot(y_train, 10)
y_test_one_hot = one_hot(y_test, 10)

activations = ['sigmoid', 'tanh', 'relu']
initializations = ['random', 'xavier', 'he']
regularizations = [False, True]
epochs = 1000
learning_rate = 0.1
results = {}

print("--- Testing Activation Functions ---")
for activation in activations:
    weight_init = 'xavier'
    l2_reg = False

    x = Input()
    n_input = X_train.shape[1]
    n_hidden = 64
    n_output = 10

    if weight_init == 'xavier':
        w1_init = np.random.randn(n_hidden, n_input) * np.sqrt(1 / n_input)
        w2_init = np.random.randn(n_output, n_hidden) * np.sqrt(1 / n_hidden)
    elif weight_init == 'he':
        w1_init = np.random.randn(n_hidden, n_input) * np.sqrt(2 / n_input)
        w2_init = np.random.randn(n_output, n_hidden) * np.sqrt(2 / n_hidden)
    else:
        w1_init = np.random.randn(n_hidden, n_input) * 0.1
        w2_init = np.random.randn(n_output, n_hidden) * 0.1

    A1 = Parameter(w1_init)
    b1 = Parameter(np.zeros((n_hidden, 1)))
    A2 = Parameter(w2_init)
    b2 = Parameter(np.zeros((n_output, 1)))
    trainables = [A1, b1, A2, b2]

    l1 = Linear(x, A1, b1)
    if activation == 'relu':
        s1 = ReLU(l1)
    elif activation == 'tanh':
        s1 = Tanh(l1)
    else:
        s1 = Sigmoid(l1)
    l2 = Linear(s1, A2, b2)
    output = Softmax(l2)

    y_true = Input()
    cost = CE(y_true, output)
    if l2_reg:
        l2_cost = L2(*trainables)
        cost = cost + l2_cost
    
    graph = topological_sort(cost)
    losses = []
    
    for i in range(epochs):
        x.forward(X_train.T)
        y_true.forward(y_train_one_hot)
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, learning_rate)
        losses.append(cost.value)

    x.forward(X_test.T)
    forward_pass(graph, final_node=output)
    predictions = np.argmax(output.value, axis=0)
    final_accuracy = np.mean(predictions == y_test)
    
    results[f'Activation: {activation}'] = (losses, final_accuracy, predictions)
    print(f"Activation: {activation}, Final Accuracy: {final_accuracy:.4f}")

print("\n--- Testing Weight Initializations ---")
for init in initializations:
    activation = 'relu'
    l2_reg = False

    x = Input()
    n_input = X_train.shape[1]
    n_hidden = 64
    n_output = 10

    if init == 'xavier':
        w1_init = np.random.randn(n_hidden, n_input) * np.sqrt(1 / n_input)
        w2_init = np.random.randn(n_output, n_hidden) * np.sqrt(1 / n_hidden)
    elif init == 'he':
        w1_init = np.random.randn(n_hidden, n_input) * np.sqrt(2 / n_input)
        w2_init = np.random.randn(n_output, n_hidden) * np.sqrt(2 / n_hidden)
    else:
        w1_init = np.random.randn(n_hidden, n_input) * 0.1
        w2_init = np.random.randn(n_output, n_hidden) * 0.1

    A1 = Parameter(w1_init)
    b1 = Parameter(np.zeros((n_hidden, 1)))
    A2 = Parameter(w2_init)
    b2 = Parameter(np.zeros((n_output, 1)))
    trainables = [A1, b1, A2, b2]

    l1 = Linear(x, A1, b1)
    if activation == 'relu':
        s1 = ReLU(l1)
    elif activation == 'tanh':
        s1 = Tanh(l1)
    else:
        s1 = Sigmoid(l1)
    l2 = Linear(s1, A2, b2)
    output = Softmax(l2)

    y_true = Input()
    cost = CE(y_true, output)
    if l2_reg:
        l2_cost = L2(*trainables)
        cost = Add(cost, l2_cost)
    
    graph = topological_sort(cost)
    losses = []

    for i in range(epochs):
        x.forward(X_train.T)
        y_true.forward(y_train_one_hot)
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, learning_rate)
        losses.append(cost.value)

    x.forward(X_test.T)
    forward_pass(graph, final_node=output)
    predictions = np.argmax(output.value, axis=0)
    final_accuracy = np.mean(predictions == y_test)

    results[f'Initialization: {init}'] = (losses, final_accuracy, predictions)
    print(f"Initialization: {init}, Final Accuracy: {final_accuracy:.4f}")

print("\n--- Testing L2 Regularization ---")
for l2 in regularizations:
    activation = 'relu'
    weight_init = 'xavier'

    x = Input()
    n_input = X_train.shape[1]
    n_hidden = 64
    n_output = 10

    if weight_init == 'xavier':
        w1_init = np.random.randn(n_hidden, n_input) * np.sqrt(1 / n_input)
        w2_init = np.random.randn(n_output, n_hidden) * np.sqrt(1 / n_hidden)
    elif weight_init == 'he':
        w1_init = np.random.randn(n_hidden, n_input) * np.sqrt(2 / n_input)
        w2_init = np.random.randn(n_output, n_hidden) * np.sqrt(2 / n_hidden)
    else:
        w1_init = np.random.randn(n_hidden, n_input) * 0.1
        w2_init = np.random.randn(n_output, n_hidden) * 0.1

    A1 = Parameter(w1_init)
    b1 = Parameter(np.zeros((n_hidden, 1)))
    A2 = Parameter(w2_init)
    b2 = Parameter(np.zeros((n_output, 1)))
    trainables = [A1, b1, A2, b2]

    l1 = Linear(x, A1, b1)
    if activation == 'relu':
        s1 = ReLU(l1)
    elif activation == 'tanh':
        s1 = Tanh(l1)
    else:
        s1 = Sigmoid(l1)
    l2 = Linear(s1, A2, b2)
    output = Softmax(l2)

    y_true = Input()
    cost = CE(y_true, output)
    if l2:
        l2_cost = L2(*trainables)
        cost = Add(cost, l2_cost)
    
    graph = topological_sort(cost)
    losses = []

    for i in range(epochs):
        x.forward(X_train.T)
        y_true.forward(y_train_one_hot)
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, learning_rate)
        losses.append(cost.value)

    x.forward(X_test.T)
    forward_pass(graph, final_node=output)
    predictions = np.argmax(output.value, axis=0)
    final_accuracy = np.mean(predictions == y_test)

    results[f'L2 Regularization: {l2}'] = (losses, final_accuracy, predictions)
    print(f"L2 Regularization: {l2}, Final Accuracy: {final_accuracy:.4f}")

for name, (losses, acc, predictions) in results.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(losses)
    axes[0].set_title("Training Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    cm = np.zeros((10, 10), dtype=int)
    for true_label, pred_label in zip(y_test, predictions):
        cm[true_label, pred_label] += 1

    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=axes[1])
    axes[1].set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=np.arange(10), yticklabels=np.arange(10),
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.suptitle(f"Experiment: {name}\nFinal Accuracy: {acc:.4f}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"task4_{name.replace(': ', '_').replace(' ', '_')}.png"
    save_plot(filename)
    plt.show()
