import numpy as np
from optical_imaging.psf import airy_psf
from optical_imaging.forward_model import forward_model
from optical_imaging.noise import add_noise
from optical_imaging.reconstruction import wiener_deconvolution
from optical_imaging.metrics import rmse

def test_snr_effect():
    obj = np.random.rand(64,64)
    psf = airy_psf(0.5)

    for photons, K in [(50, 0.1), (1000, 0.001)]:
        y = add_noise(forward_model(obj, psf), photons=photons)
        x_hat = wiener_deconvolution(y, psf, K=K)

        # Physical invariants
        assert np.isfinite(x_hat).all()
        assert x_hat.shape == obj.shape