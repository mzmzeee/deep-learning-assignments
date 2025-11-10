import numpy as np
from funxs import im2col, col2im

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

        FN, C, FH, FW = W.shape
        N, _, H, W_in = x.shape
        stride = self.stride
        pad = self.padding

        out_h = (H + 2 * pad - FH) // stride + 1
        out_w = (W_in + 2 * pad - FW) // stride + 1

        col = im2col(x, FH, FW, stride, pad)
        col_W = W.reshape(FN, -1).T

        out = col @ col_W + b
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        self.value = out
        self.cache = (col, x.shape)

    def backward(self):
        col, x_shape = self.cache
        W = self.inputs[1].value
        FN, C, FH, FW = W.shape
        stride = self.stride
        pad = self.padding

        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        col_W = W.reshape(FN, -1).T

        for n in self.outputs:
            grad_cost = n.gradients[self]
            grad_cost = grad_cost.transpose(0, 2, 3, 1).reshape(-1, FN)

            self.gradients[self.inputs[2]] += np.sum(grad_cost, axis=0)

            dW = col.T @ grad_cost
            dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)
            self.gradients[self.inputs[1]] += dW

            dcol = grad_cost @ col_W.T
            dx = col2im(dcol, x_shape, FH, FW, stride, pad)
            self.gradients[self.inputs[0]] += dx

class MaxPooling(Node):
    def __init__(self, input_node, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        Node.__init__(self, [input_node])

    def forward(self):
        x = self.inputs[0].value
        N, C, H, W = x.shape
        pool_size = self.pool_size
        stride = self.stride

        out_h = (H - pool_size) // stride + 1
        out_w = (W - pool_size) // stride + 1

        col = im2col(x, pool_size, pool_size, stride)
        col = col.reshape(N * out_h * out_w, C, pool_size * pool_size)

        arg_max = np.argmax(col, axis=2)
        out = np.max(col, axis=2)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.value = out
        self.cache = (arg_max, x.shape, out_h, out_w)

    def backward(self):
        arg_max, x_shape, out_h, out_w = self.cache
        pool_size = self.pool_size
        stride = self.stride
        N, C, _, _ = x_shape

        self.gradients = {self.inputs[0]: np.zeros(x_shape, dtype=self.value.dtype)}

        for n in self.outputs:
            grad_cost = n.gradients[self]
            grad_cost = grad_cost.transpose(0, 2, 3, 1)
            grad_flat = grad_cost.reshape(-1, C)

            dcol = np.zeros((grad_flat.shape[0], C, pool_size * pool_size), dtype=grad_flat.dtype)
            flat_dcol = dcol.reshape(-1, pool_size * pool_size)
            flat_indices = arg_max.reshape(-1)
            flat_values = grad_flat.reshape(-1)
            flat_dcol[np.arange(flat_indices.size), flat_indices] = flat_values

            dcol = dcol.reshape(-1, C * pool_size * pool_size)
            dx = col2im(dcol, x_shape, pool_size, pool_size, stride)
            self.gradients[self.inputs[0]] += dx

class Flatten(Node):
    def __init__(self, input_node):
        Node.__init__(self, [input_node])

    def forward(self):
        x = self.inputs[0].value
        self.orig_shape = x.shape
        batch_size = x.shape[0]
        self.value = x.reshape(batch_size, -1).T

    def backward(self):
        self.gradients = {self.inputs[0]: np.zeros_like(self.inputs[0].value)}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            reshaped = grad_cost.T.reshape(self.orig_shape)
            self.gradients[self.inputs[0]] += reshaped

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

