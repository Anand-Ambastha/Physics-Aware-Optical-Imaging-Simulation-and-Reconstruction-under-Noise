import numpy as np
from scipy.special import j1
from optical_imaging.config import WAVELENGTH, PIXEL_SIZE, IMAGE_SIZE

def airy_psf(na):
    k = 2 * np.pi / WAVELENGTH
    coords = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * PIXEL_SIZE
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2) + 1e-12

    psf = (2 * j1(k * na * R) / (k * na * R))**2
    psf /= psf.sum()
    return psf
