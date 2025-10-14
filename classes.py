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
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])
    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.mean(-y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value))
    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (y_pred.value - y_true.value) / y_true.value.shape[1]
        self.gradients[y_true] = 0