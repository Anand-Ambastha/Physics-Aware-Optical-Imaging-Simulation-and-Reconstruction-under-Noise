import numpy as np

WAVELENGTH = 550e-9      # meters (Goodman, Fourier Optics)
PIXEL_SIZE = 100e-9     # meters
IMAGE_SIZE = 256

NA_VALUES = np.linspace(0.1, 0.8, 8)

READ_NOISE_STD = 2.0    # electrons RMS (Janesick)
PHOTON_LEVELS = [50, 100, 500, 1000]
