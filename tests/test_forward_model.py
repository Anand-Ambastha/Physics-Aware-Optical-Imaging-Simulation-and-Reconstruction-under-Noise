import numpy as np
from optical_imaging.forward_model import forward_model

def test_linearity():
    x1 = np.random.rand(64,64)
    x2 = np.random.rand(64,64)
    psf = np.zeros((64,64))
    psf[32,32] = 1.0
    assert np.allclose(
        forward_model(x1 + x2, psf),
        forward_model(x1, psf) + forward_model(x2, psf)
    )
