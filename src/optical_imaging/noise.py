import numpy as np
from optical_imaging.config import READ_NOISE_STD


def add_noise(image, photons):
    image = np.asarray(image, dtype=np.float64)

    eps = 1e-12
    max_val = image.max()

    if max_val < eps:
        scaled = np.zeros_like(image)
    else:
        scaled = image / max_val * photons

    # Physical safety: non-negative, finite photon budget
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(scaled, 0.0, 1e6)

    noisy = np.random.poisson(scaled).astype(np.float64)
    noisy += np.random.normal(0, READ_NOISE_STD, noisy.shape)

    return noisy
