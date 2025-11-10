from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def im2col(x, filter_h, filter_w, stride=1, pad=0):
    """Roll sliding window blocks into columns for fast convolution."""
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode="constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=x.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride * out_w
            col[:, :, y, x_idx, :, :] = img[:, :, y:y_max:stride, x_idx:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Reverse im2col, scattering columns back into the source image shape."""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride * out_w
            img[:, :, y:y_max:stride, x_idx:x_max:stride] += col[:, :, y, x_idx, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

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
        
def gen_xordata(samples = 100 ,test_percent = 0.3, noise=0.1 ):
    MEAN1 = np.array([0, 0])
    MEAN2 = np.array([5, 5])
    MEAN3 = np.array([0, 5])
    MEAN4 = np.array([5, 0])

    COV = np.array([[noise, 0], [0, noise]])

    X1 = multivariate_normal.rvs(mean=MEAN1, cov=COV, size=samples)
    X2 = multivariate_normal.rvs(mean=MEAN2, cov=COV, size=samples)
    X3 = multivariate_normal.rvs(mean=MEAN3, cov=COV, size=samples)
    X4 = multivariate_normal.rvs(mean=MEAN4, cov=COV, size=samples)

    X_class0 = np.vstack((X1, X2))
    y_class0 = np.zeros(len(X_class0))

    X_class1 = np.vstack((X3, X4))
    y_class1 = np.ones(len(X_class1))

    X = np.vstack((X_class0, X_class1))
    y = np.hstack((y_class0, y_class1))

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(len(X) * test_percent)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def split_data(X, y, test_size=0.25, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def topological_sort(entry_node):
    visited = set()
    sorted_nodes = []

    def visit(node):
        if node not in visited:
            visited.add(node)
            for n in node.inputs:
                visit(n)
            sorted_nodes.append(node)

    visit(entry_node)
    return sorted_nodes

def get_trainable(graph):
    from classes import Parameter
    trainable_nodes = []
    for node in graph:
        if isinstance(node, Parameter):
            trainable_nodes.append(node)
    return trainable_nodes

def _ensure_figure_dir():
    path = 'assingment1/figure'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_plot(filename, fig=None):
    path = _ensure_figure_dir()
    output_path = os.path.join(path, filename)
    if fig is None:
        plt.savefig(output_path)
    else:
        fig.savefig(output_path)

def one_hot(y, num_classes):
    y_one_hot = np.zeros((num_classes, y.shape[0]))
    y_one_hot[y, np.arange(y.shape[0])] = 1
    return y_one_hot

def plot_decision_boundary(pred_func, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    ax.set_title(title)


def _serialise_metric_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return value.tolist()
    return value


def log_metrics(task_name, metrics):
    metrics_path = os.path.join(_ensure_figure_dir(), 'c_task_metrics.json')
    stored = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as handle:
            try:
                stored = json.load(handle)
            except json.JSONDecodeError:
                stored = {}

    processed = {key: _serialise_metric_value(value) for key, value in metrics.items()}
    stored[task_name] = processed

    with open(metrics_path, 'w') as handle:
        json.dump(stored, handle, indent=2)


def load_logged_metrics():
    metrics_path = os.path.join(_ensure_figure_dir(), 'c_task_metrics.json')
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, 'r') as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def plot_metric_summary(metric_key='accuracy', filename='c_tasks_accuracy.png', title=None):
    metrics = load_logged_metrics()
    labels = []
    values = []

    for task in sorted(metrics.keys()):
        raw_value = metrics[task].get(metric_key)
        if raw_value is None:
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        labels.append(task)
        values.append(numeric_value)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color='#4c72b0')
    ax.set_ylabel(metric_key.replace('_', ' ').title())
    upper = max(1.0, max(values) + 0.05)
    ax.set_ylim(0, upper)
    ax.set_title(title or f'{metric_key.replace("_", " ").title()} Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    save_plot(filename, fig=fig)
    plt.close(fig)


def get_best_metric(metric_key='accuracy'):
    metrics = load_logged_metrics()
    best_task = None
    best_value = None
    for task, task_metrics in metrics.items():
        raw_value = task_metrics.get(metric_key)
        if raw_value is None:
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if best_value is None or numeric_value > best_value:
            best_task = task
            best_value = numeric_value
    return best_task, best_value


def plot_sample_predictions(images, y_true, y_pred, filename, num_samples=16, class_names=None, random_state=0):
    total = len(y_true)
    if total == 0:
        return
    num_samples = min(num_samples, total)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(total, size=num_samples, replace=False)

    cols = min(4, num_samples)
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    for ax, idx in zip(axes_flat, indices):
        image = images[idx]
        if image.ndim == 3 and image.shape[0] in (1, 3):
            if image.shape[0] == 1:
                ax.imshow(image[0], cmap='gray')
            else:
                ax.imshow(np.moveaxis(image, 0, -1))
        else:
            ax.imshow(image, cmap='gray')
        pred = y_pred[idx]
        true = y_true[idx]
        pred_label = class_names[pred] if class_names is not None else pred
        title = f'Pred {pred_label}'
        if pred != true:
            true_label = class_names[true] if class_names is not None else true
            title += f' | True {true_label}'
            ax.set_title(title, color='red', fontsize=8)
        else:
            ax.set_title(title, color='green', fontsize=8)
        ax.axis('off')

    for ax in axes_flat[num_samples:]:
        ax.axis('off')

    fig.suptitle('Sample Predictions', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    save_plot(filename, fig=fig)
    plt.close(fig)


def plot_feature_maps(feature_maps, filename, title='', max_maps=8, cmap='viridis'):
    if feature_maps.ndim != 4:
        return
    sample = feature_maps[0]
    num_maps = min(sample.shape[0], max_maps)
    cols = min(4, num_maps)
    rows = int(np.ceil(num_maps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_2d(axes)

    for idx, ax in enumerate(axes.flat):
        if idx >= num_maps:
            ax.axis('off')
            continue
        ax.imshow(sample[idx], cmap=cmap)
        ax.set_title(f'Map {idx}', fontsize=8)
        ax.axis('off')

    fig.suptitle(title or 'Feature Maps', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    save_plot(filename, fig=fig)
    plt.close(fig)


def compute_classification_metrics(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    accuracy = float(tp.sum() / np.maximum(cm.sum(), 1))

    metrics = {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'accuracy': accuracy,
        'macro_precision': float(np.mean(precision)),
        'macro_recall': float(np.mean(recall)),
        'macro_f1': float(np.mean(f1))
    }
    return metrics


def format_classification_metrics(metrics, class_names=None):
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    support = metrics['support']
    num_classes = precision.shape[0]

    lines = []
    header = f"{'Class':>8}  {'Precision':>9}  {'Recall':>7}  {'F1':>5}  {'Support':>8}"
    lines.append(header)
    lines.append('-' * len(header))
    for idx in range(num_classes):
        name = class_names[idx] if class_names is not None else idx
        lines.append(f"{str(name):>8}  {precision[idx]:9.3f}  {recall[idx]:7.3f}  {f1[idx]:5.3f}  {int(support[idx]):8d}")

    lines.append('-' * len(header))
    lines.append(f"{'Macro':>8}  {metrics['macro_precision']:9.3f}  {metrics['macro_recall']:7.3f}  {metrics['macro_f1']:5.3f}  {int(np.sum(support)):8d}")
    lines.append(f"{'Accuracy':>8}  {metrics['accuracy']:9.3f}")
    return '\n'.join(lines)