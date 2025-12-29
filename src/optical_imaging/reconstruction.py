import numpy as np

def wiener_deconvolution(y, psf, K):
    Y = np.fft.fft2(y)
    H = np.fft.fft2(psf, s=y.shape)
    Hc = np.conj(H)
    X_hat = Hc / (np.abs(H)**2 + K) * Y
    return np.real(np.fft.ifft2(X_hat))
