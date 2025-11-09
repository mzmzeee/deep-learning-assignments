import numpy as np

class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}
        for node in inputs:
            node.outputs.append(self)
    def forward(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

class Linear(Node):
    def __init__(self, x, A, b):
        Node.__init__(self, inputs=[x, A, b])
    def forward(self):
        x = self.inputs[0].value
        A = self.inputs[1].value
        b = self.inputs[2].value
        self.value = A @ x + b
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            x = self.inputs[0]
            A = self.inputs[1]
            b = self.inputs[2]
            self.gradients[x] += A.value.T @ grad_cost
            self.gradients[A] += grad_cost @ x.value.T
            self.gradients[b] += np.sum(grad_cost, axis=1, keepdims=True)

class AutomatedLinear(Node):
    def __init__(self, x, n_in, n_out):
        self.A = Parameter(np.random.randn(n_out, n_in) * 0.1)
        self.b = Parameter(np.zeros((n_out, 1)))
        Node.__init__(self, inputs=[x, self.A, self.b])

    def forward(self):
        x = self.inputs[0].value
        A = self.inputs[1].value
        b = self.inputs[2].value
        self.value = A @ x + b
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            x = self.inputs[0]
            A = self.inputs[1]
            b = self.inputs[2]
            self.gradients[x] += A.value.T @ grad_cost
            self.gradients[A] += grad_cost @ x.value.T
            self.gradients[b] += np.sum(grad_cost, axis=1, keepdims=True)

class Input(Node):
    def __init__(self):
        Node.__init__(self)
    def forward(self, value=None):
        if value is not None:
            self.value = value
    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value
    def forward(self):
        pass
    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)
    def backward(self):
        self.gradients = {self.inputs[0]: np.zeros_like(self.inputs[0].value)}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            partial = self.value * (1 - self.value)
            self.gradients[self.inputs[0]] += grad_cost * partial

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])
    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.mean(-y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value))
    def backward(self):
        y_true, y_pred = self.inputs
        grad_y_pred = (y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value))
        self.gradients[y_pred] = grad_y_pred / y_true.value.shape[1]
        self.gradients[y_true] = np.zeros_like(y_true.value)

class Softmax(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value = self.inputs[0].value
        exps = np.exp(input_value - np.max(input_value, axis=0, keepdims=True))
        self.value = exps / np.sum(exps, axis=0, keepdims=True)

    def backward(self):
        self.gradients = {self.inputs[0]: np.zeros_like(self.inputs[0].value)}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] += grad_cost

class CE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        epsilon = 1e-9
        self.value = -np.sum(y_true.value * np.log(y_pred.value + epsilon)) / y_true.value.shape[1]

    def backward(self):
        y_true, y_pred = self.inputs
        grad_y_pred = y_pred.value - y_true.value
        self.gradients = {
            y_pred: grad_y_pred / y_true.value.shape[1],
            y_true: np.zeros_like(y_true.value)
            }

class Tanh(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        self.value = np.tanh(self.inputs[0].value)

    def backward(self):
        self.gradients = {self.inputs[0]: np.zeros_like(self.inputs[0].value)}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] += grad_cost * (1 - self.value**2)

class ReLU(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        self.value = np.maximum(0, self.inputs[0].value)

    def backward(self):
        self.gradients = {self.inputs[0]: np.zeros_like(self.inputs[0].value)}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] += grad_cost * (self.inputs[0].value > 0)

class Conv(Node):
    def __init__(self, input_node, num_filters, input_channels, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        weight_shape = (num_filters, input_channels, filter_size, filter_size)
        weight_scale = np.sqrt(2.0 / (input_channels * filter_size * filter_size))
        self.W = Parameter(np.random.randn(*weight_shape) * weight_scale)
        self.b = Parameter(np.zeros(num_filters))

        Node.__init__(self, [input_node, self.W, self.b])

    def forward(self):
        x = self.inputs[0].value
        W = self.inputs[1].value
        b = self.inputs[2].value

        batch_size, _, height, width = x.shape
        pad = self.padding
        stride = self.stride
        filter_size = self.filter_size

        if pad > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            x_padded = x

        out_height = (x_padded.shape[2] - filter_size) // stride + 1
        out_width = (x_padded.shape[3] - filter_size) // stride + 1

        out = np.zeros((batch_size, self.num_filters, out_height, out_width))

        for i in range(out_height):
            h_start = i * stride
            h_end = h_start + filter_size
            for j in range(out_width):
                w_start = j * stride
                w_end = w_start + filter_size
                region = x_padded[:, :, h_start:h_end, w_start:w_end]
                for f in range(self.num_filters):
                    out[:, f, i, j] = np.sum(region * W[f][None, :, :, :], axis=(1, 2, 3)) + b[f]

        self.value = out
        self.cache = (x, x_padded)

    def backward(self):
        x, x_padded = self.cache
        W = self.inputs[1].value
        pad = self.padding
        stride = self.stride
        filter_size = self.filter_size

        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(W)
        db = np.zeros_like(self.inputs[2].value)

        out_height = self.value.shape[2]
        out_width = self.value.shape[3]

        for n in self.outputs:
            grad_cost = n.gradients[self]
            db += np.sum(grad_cost, axis=(0, 2, 3))
            for i in range(out_height):
                h_start = i * stride
                h_end = h_start + filter_size
                for j in range(out_width):
                    w_start = j * stride
                    w_end = w_start + filter_size
                    region = x_padded[:, :, h_start:h_end, w_start:w_end]
                    for f in range(self.num_filters):
                        grad = grad_cost[:, f, i, j][:, None, None, None]
                        dW[f] += np.sum(region * grad, axis=0)
                        dx_padded[:, :, h_start:h_end, w_start:w_end] += grad * W[f][None, :, :, :]

        if pad > 0:
            dx = dx_padded[:, :, pad:-pad, pad:-pad]
        else:
            dx = dx_padded

        self.gradients = {
            self.inputs[0]: dx,
            self.inputs[1]: dW,
            self.inputs[2]: db
        }

class MaxPooling(Node):
    def __init__(self, input_node, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        Node.__init__(self, [input_node])

    def forward(self):
        x = self.inputs[0].value
        batch_size, channels, height, width = x.shape
        pool_size = self.pool_size
        stride = self.stride

        out_height = (height - pool_size) // stride + 1
        out_width = (width - pool_size) // stride + 1

        out = np.zeros((batch_size, channels, out_height, out_width))
        self.max_mask = np.zeros_like(x, dtype=bool)

        for i in range(out_height):
            h_start = i * stride
            h_end = h_start + pool_size
            for j in range(out_width):
                w_start = j * stride
                w_end = w_start + pool_size
                window = x[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(window, axis=(2, 3))
                out[:, :, i, j] = max_vals
                mask = window == max_vals[:, :, None, None]
                self.max_mask[:, :, h_start:h_end, w_start:w_end] = mask

        self.value = out

    def backward(self):
        x = self.inputs[0].value
        pool_size = self.pool_size
        stride = self.stride
        out_height = self.value.shape[2]
        out_width = self.value.shape[3]

        dx = np.zeros_like(x)

        for n in self.outputs:
            grad_cost = n.gradients[self]
            for i in range(out_height):
                h_start = i * stride
                h_end = h_start + pool_size
                for j in range(out_width):
                    w_start = j * stride
                    w_end = w_start + pool_size
                    window_mask = self.max_mask[:, :, h_start:h_end, w_start:w_end]
                    grad = grad_cost[:, :, i, j][:, :, None, None]
                    dx[:, :, h_start:h_end, w_start:w_end] += grad * window_mask

        self.gradients = {self.inputs[0]: dx}

class L2(Node):
    def __init__(self, *params):
        Node.__init__(self, params)
        self.l2_lambda = 1.0

    def forward(self):
        self.value = 0
        for param in self.inputs:
            self.value += np.sum(param.value**2)
        self.value *= self.l2_lambda / (2 * self.inputs[0].value.shape[1])

    def backward(self):
        for param in self.inputs:
            self.gradients[param] = self.l2_lambda * param.value / self.inputs[0].value.shape[1]

class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum(n.value for n in self.inputs)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            for input_node in self.inputs:
                self.gradients[input_node] += grad_cost

