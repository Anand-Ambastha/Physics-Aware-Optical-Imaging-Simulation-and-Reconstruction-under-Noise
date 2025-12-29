import numpy as np
from optical_imaging.noise import add_noise

def test_noise_variance():
    img = np.ones((64,64))
    noisy = add_noise(img, photons=100)
    assert noisy.var() > img.var()
