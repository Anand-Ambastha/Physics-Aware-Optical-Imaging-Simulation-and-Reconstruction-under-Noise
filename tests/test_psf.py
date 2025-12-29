import numpy as np
from optical_imaging.psf import airy_psf

def test_psf_energy():
    psf = airy_psf(0.5)
    assert abs(psf.sum() - 1.0) < 1e-3

def test_psf_non_negative():
    psf = airy_psf(0.5)
    assert (psf >= 0).all()
