import numpy as np


def softmax(x):
    e_x = np.exp(x)
    div = np.repeat(np.expand_dims(np.sum(e_x, axis=1), 1), 5, axis=1)
    return np.divide(e_x, div)


def myprint(string, *args):
    silent = True
    silent = False
    if not silent:
        print(string, *args)  # print(*args) - also works if we goto myprint(*args)


