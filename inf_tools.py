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


def rolling_window_nodelay(vec, window, step):
    def calculate_padding(vec, window, step):
        import math
        N = len(vec)
        B = math.ceil(N / step)
        L = (B - 1) * step + window
        return L - N
    from skimage.util import view_as_windows

    pad = calculate_padding(vec, window, step)
    A = view_as_windows(np.pad(vec, (0, pad)), window, step).T
    zero_cols = pad // step
    return np.delete(A, np.arange(A.shape[1] - zero_cols, A.shape[1]), axis=1)
