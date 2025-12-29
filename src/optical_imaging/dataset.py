import numpy as np

def generate_object(size):
    x = np.zeros((size, size))
    x[64:192, 96:160] = 1.0
    x[96:160, 64:192] += 0.5
    return x
