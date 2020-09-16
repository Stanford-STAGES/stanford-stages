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
    from skimage.util import view_as_windows
    n = len(vec)
    pad = (window-n) % step

    # Only happens if we our window is greater than our vector, in which case we have problems anyway.
    if pad < 0:
        pad = 0
    return view_as_windows(np.pad(vec, (0, pad)), window, step).T

'''
To be removed after 10/10/2020:
def calculate_padding(vec, window, step):
import math
N = len(vec)
B = math.ceil(N / step)  # perhaps B = (N-window)//step + 1
L = (B - 1) * step + window
return L - N
# However, the above does not hold for
# vec = [1 2 3 4 5], window = 4, step =2, n = len(vec)
# buffer(vec, window, 4, 4-2) results in 0x1
# The above results in N=5, B=3, L=8, padding 0x3

# Mathematically this: window - step if N/step is an integer.
# otherwise it is:  window - step + (step - N % step) = window - N % step
# window - N % step     
'''


