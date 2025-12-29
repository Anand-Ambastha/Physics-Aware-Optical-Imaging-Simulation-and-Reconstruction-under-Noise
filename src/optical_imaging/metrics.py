import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def rmse(x, x_hat):
    return np.sqrt(np.mean((x - x_hat)**2))

def compute_psnr(x, x_hat):
    return psnr(x, x_hat, data_range=x.max() - x.min())

def nrmse(x, x_hat):
    return rmse(x, x_hat) / (np.linalg.norm(x) + 1e-12)
