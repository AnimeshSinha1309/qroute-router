import numpy as np


def histogram(array, max_val, min_val=0):
    hist = np.zeros(max_val - min_val + 1, dtype=np.int32)
    for val in array:
        hist[val - min_val] += 1
    return hist
