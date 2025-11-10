import numpy as np
from classes import *
from funxs import *

sample_image = np.zeros((1, 1, 8, 8), dtype=np.float32)
sample_image[0, 0, 2:6, 3] = 1.0
sample_image[0, 0, 2, 2:5] = 1.0
sample_image[0, 0, 5, 2:5] = 1.0
sample_label = "synthetic-0"
print(f"Inspecting synthetic pattern with label {sample_label}")

x = Input()
conv_layer = Conv(x, num_filters=2, input_channels=1, filter_size=3, stride=1, padding=1)
pool_layer = MaxPooling(conv_layer, pool_size=2, stride=2)

x.forward(sample_image)
conv_graph = topological_sort(conv_layer)
forward_pass(conv_graph)
print("Conv output shape:", conv_layer.value.shape)
print("Conv feature map 0 (rounded):\n", np.round(conv_layer.value[0, 0], 3))

pool_graph = topological_sort(pool_layer)
forward_pass(pool_graph)
print("MaxPool output shape:", pool_layer.value.shape)
print("MaxPool feature map 0 (rounded):\n", np.round(pool_layer.value[0, 0], 3))

plot_feature_maps(sample_image, 'c_task2_input_digit.png', title=f'Input Pattern {sample_label}', max_maps=1, cmap='gray')
plot_feature_maps(conv_layer.value, 'c_task2_conv_maps.png', title='Conv Layer Feature Maps', max_maps=4, cmap='magma')
plot_feature_maps(pool_layer.value, 'c_task2_pool_maps.png', title='MaxPool Feature Maps', max_maps=4, cmap='magma')

log_metrics('c_task2', {
    'conv_filters': conv_layer.value.shape[1],
    'conv_height': conv_layer.value.shape[2],
    'conv_width': conv_layer.value.shape[3],
    'pool_height': pool_layer.value.shape[2],
    'pool_width': pool_layer.value.shape[3]
})

best_task, best_value = get_best_metric('accuracy')
if best_task is not None and best_value is not None:
	print(f"Best recorded classification accuracy so far: {best_task} at {best_value:.2%}")
