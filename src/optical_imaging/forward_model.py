from scipy.signal import fftconvolve

def forward_model(obj, psf):
    return fftconvolve(obj, psf, mode="same")
