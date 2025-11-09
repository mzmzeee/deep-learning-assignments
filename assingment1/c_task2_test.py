import numpy as np
from classes import *
from funxs import *

sample_input = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)

x = Input()
conv_layer = Conv(x, num_filters=2, input_channels=1, filter_size=3, stride=1, padding=1)
pool_layer = MaxPooling(conv_layer, pool_size=2, stride=2)

x.forward(sample_input)
conv_graph = topological_sort(conv_layer)
forward_pass(conv_graph)
print("Conv output shape:", conv_layer.value.shape)

pool_graph = topological_sort(pool_layer)
forward_pass(pool_graph)
print("MaxPool output shape:", pool_layer.value.shape)
