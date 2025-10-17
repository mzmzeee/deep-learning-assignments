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