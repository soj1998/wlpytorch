import numpy as np


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

print([0]*6 + [0, 1, 0, 0])
ret = [int(i) for i in '{0:b}'.format(10)]
print(ret)
for i in '{0:b}'.format(10):
    print(i)