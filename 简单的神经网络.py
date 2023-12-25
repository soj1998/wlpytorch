import numpy as np
import torch

def binary_encoder(input_size):
    def wrapper(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (input_size - len(ret)) + ret
    return wrapper


def get_numpy_data(input_size=10, limit=1000):
    x = []
    y = []
    encoder = binary_encoder(input_size)
    for i in range(limit):
        x.append(encoder(i))
        if i % 15 == 0:
            y.append([1, 0, 0, 0])
        elif i % 5 == 0:
            y.append([0, 1, 0, 0])
        elif i % 3 == 0:
            y.append([0, 0, 1, 0])
        else:
            y.append([0, 0, 1, 0])
    # return training_test_gen(np.array(x), np.array(y))

epochs = 500
batches = 64
lr = 0.01
input_size = 10
output_size = 4
hidden_size = 100
# for i in epoch:
#     network_execution_over_whole_dataset()
x = torch.from_numpy(trX).type(dtype)
y = torch.from_numpy(trY).type(dtype)
w1 = torch.randn(input_size, hidden_size, requires_grad=True).type(dtype)
w2 = torch.randn(hidden_size, output_size, requires_grad=True).type(dtype)
b1 = torch.zeros(1, hidden_size, requires_grad=True).type(dtype)
b2 = torch.zeros(1, output_size, requires_grad=True).type(dtype)

for epoch in range(epochs):
    for batch in range(no_of_batches):
        start = batch* batches
        end = start + batches
        x_ = x[start:end]
        y_ = y[start:end]

        #building graph
        a2 = x_.matmul(w1)
        a2 = a2.add(b1)
        print(a2.grad, a2.grad_fn,a2)

        #None <AddBackward0 object at 0x7f5f3b9253c8> tensor([[...]])
        h2 = a2.sigmoid()
        a3 = h2.matmul(w2)
        a3=a3.add(b2)
        hyp =a3.sigmoid()
        error = hyp-y_
        output = error.pow(2).sum()/2.0

        # backpropagation
        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()
        output.backward()

        print(x.grad, x.grad_fn, x)
        # None None tensor([[...]])
        print(w1.grad, w1.grad_fn, w1)
        # tensor([[...]], None, tensor([[...]]))
        print(a2.grad, a2.grad_fn, a2)
        # None <AddBackward 0 object at 0x7f5f3d42c780> tensor([[...]])

        # parameter update
        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad
            b1 -= lr * b1.grad
            b2 -= lr * b2.grad