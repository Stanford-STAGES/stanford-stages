import numpy as np
import logging
from pathlib import Path
from pyedflib import EdfReader

logging_level_STAGES = 60
logging.addLevelName(logging_level_STAGES, 'STAGES')
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def print_log(msg: str, log_level: str = 'info'):
    print(msg)
    if log_level in ['warning', 'debug', 'critical', 'error', 'info']:
        getattr(logger, log_level)(msg)
    elif log_level.lower() == 'stages':
        logger.log(logging_level_STAGES, msg)


def get_edf_filenames(path2check):
    edf_files = get_edf_files(path2check)
    return [str(i) for i in edf_files]  # list compression


def get_edf_files(path2check):
    p = Path(path2check)
    # verify that we have an accurate directory
    # if so then list all .edf/.EDF files
    if p.is_dir():
        print_log('Checking ' + str(path2check) + "for edf files.", 'debug')
        edf_files = p.glob('*.[Ee][Dd][Ff]')  # make search case-insensitive
    else:
        print_log(str(path2check) + " is not a valid directory.", 'debug')
        edf_files = []
    return list(edf_files)


def get_signal_headers(edf_filename, verbose=False):
    if verbose:
        print("Reading headers from ", edf_filename)
    try:
        edf_r = EdfReader(str(edf_filename), annotations_mode=False, check_file_size=False)
        return edf_r.getSignalHeaders()
    except:
        print("Failed reading headers from ", str(edf_filename))
        return []


def get_channel_labels(edf_filename):
    channel_headers = get_signal_headers(edf_filename)
    return [fields["label"] for fields in channel_headers]


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
    from skimage.util import view_as_windows

    n = len(vec)
    pad = (window-n) % step
    # if n % step == 0:
    #    pad = window - step
    #else:
    #    pad = window - n % step

    # insanity check - only happens if we our window is greater than our vector, in which case we'll have problems
    # anyway..
    if pad < 0:
        pad = 0

    # pad = calculate_padding(vec, window, step)
    A = view_as_windows(np.pad(vec, (0, pad)), window, step).T

#    pad = calculate_padding(vec, window, step)
#    A = view_as_windows(np.pad(vec, (0, pad)), window, step).T

#    zero_cols = pad // step
#    A = np.delete(A, np.arange(A.shape[1] - zero_cols, A.shape[1]), axis=1)
    return A

