import os
from textwrap import dedent

FILES = {
    "requirements.txt": """
numpy
scipy
matplotlib
torch
torchvision
scikit-image
pytest
""",

    "config.py": """
import numpy as np

WAVELENGTH = 550e-9      # meters (Goodman, Fourier Optics)
PIXEL_SIZE = 100e-9     # meters
IMAGE_SIZE = 256

NA_VALUES = np.linspace(0.1, 0.8, 8)

READ_NOISE_STD = 2.0    # electrons RMS (Janesick)
PHOTON_LEVELS = [50, 100, 500, 1000]
""",

    "psf.py": """
import numpy as np
from scipy.special import j1
from config import WAVELENGTH, PIXEL_SIZE, IMAGE_SIZE

def airy_psf(na):
    k = 2 * np.pi / WAVELENGTH
    coords = (np.arange(IMAGE_SIZE) - IMAGE_SIZE // 2) * PIXEL_SIZE
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2) + 1e-12

    psf = (2 * j1(k * na * R) / (k * na * R))**2
    psf /= psf.sum()
    return psf
""",

    "forward_model.py": """
from scipy.signal import fftconvolve

def forward_model(obj, psf):
    return fftconvolve(obj, psf, mode="same")
""",

    "noise.py": """
import numpy as np
from config import READ_NOISE_STD

def add_noise(image, photons):
    scaled = image / image.max() * photons
    noisy = np.random.poisson(scaled)
    noisy += np.random.normal(0, READ_NOISE_STD, noisy.shape)
    return noisy
""",

    "reconstruction.py": """
import numpy as np

def wiener_deconvolution(y, psf, K):
    Y = np.fft.fft2(y)
    H = np.fft.fft2(psf, s=y.shape)
    Hc = np.conj(H)
    X_hat = Hc / (np.abs(H)**2 + K) * Y
    return np.real(np.fft.ifft2(X_hat))
""",

    "metrics.py": """
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def rmse(x, x_hat):
    return np.sqrt(np.mean((x - x_hat)**2))

def compute_psnr(x, x_hat):
    return psnr(x, x_hat, data_range=x.max() - x.min())
""",

    "dataset.py": """
import numpy as np

def generate_object(size):
    x = np.zeros((size, size))
    x[64:192, 96:160] = 1.0
    x[96:160, 64:192] += 0.5
    return x
""",

    "cnn.py": """
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
""",

    "experiments/resolution_vs_na.py": """
import matplotlib.pyplot as plt
from config import NA_VALUES, WAVELENGTH

cutoffs = [2 * na / WAVELENGTH for na in NA_VALUES]

plt.plot(NA_VALUES, cutoffs)
plt.xlabel("Numerical Aperture")
plt.ylabel("Cutoff Frequency (cycles/m)")
plt.title("Resolution vs NA (Diffraction Limit)")
plt.show()
""",

    "experiments/snr_sweep.py": """
from dataset import generate_object
from psf import airy_psf
from forward_model import forward_model
from noise import add_noise
from reconstruction import wiener_deconvolution
from metrics import rmse
from config import IMAGE_SIZE, PHOTON_LEVELS

obj = generate_object(IMAGE_SIZE)
psf = airy_psf(0.5)

for photons in PHOTON_LEVELS:
    y = add_noise(forward_model(obj, psf), photons)
    x_hat = wiener_deconvolution(y, psf, K=0.01)
    print(f"Photons: {photons}, RMSE: {rmse(obj, x_hat):.4f}")
""",

    "experiments/failure_modes.py": """
import matplotlib.pyplot as plt
from dataset import generate_object
from psf import airy_psf
from forward_model import forward_model
from noise import add_noise
from reconstruction import wiener_deconvolution

obj = generate_object(256)
psf = airy_psf(0.7)
y = add_noise(forward_model(obj, psf), photons=20)
x_hat = wiener_deconvolution(y, psf, K=0.001)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Ground Truth"); plt.imshow(obj); plt.axis("off")
plt.subplot(1,3,2); plt.title("Measurement"); plt.imshow(y); plt.axis("off")
plt.subplot(1,3,3); plt.title("Reconstruction"); plt.imshow(x_hat); plt.axis("off")
plt.show()
""",

    "tests/test_psf.py": """
import numpy as np
from psf import airy_psf

def test_psf_energy():
    psf = airy_psf(0.5)
    assert abs(psf.sum() - 1.0) < 1e-3

def test_psf_non_negative():
    psf = airy_psf(0.5)
    assert (psf >= 0).all()
""",

    "tests/test_forward_model.py": """
import numpy as np
from forward_model import forward_model

def test_linearity():
    x1 = np.random.rand(64,64)
    x2 = np.random.rand(64,64)
    psf = np.zeros((64,64))
    psf[32,32] = 1.0
    assert np.allclose(
        forward_model(x1 + x2, psf),
        forward_model(x1, psf) + forward_model(x2, psf)
    )
""",

    "tests/test_noise.py": """
import numpy as np
from noise import add_noise

def test_noise_variance():
    img = np.ones((64,64))
    noisy = add_noise(img, photons=100)
    assert noisy.var() > img.var()
""",

    "tests/test_reconstruction.py": """
import numpy as np
from psf import airy_psf
from forward_model import forward_model
from noise import add_noise
from reconstruction import wiener_deconvolution
from metrics import rmse

def test_snr_effect():
    obj = np.random.rand(64,64)
    psf = airy_psf(0.5)
    low = add_noise(forward_model(obj, psf), photons=50)
    high = add_noise(forward_model(obj, psf), photons=1000)
    assert rmse(obj, wiener_deconvolution(high, psf, 0.01)) < \
           rmse(obj, wiener_deconvolution(low, psf, 0.01))
""",

    "tests/test_metrics.py": """
import numpy as np
from metrics import rmse

def test_rmse_zero():
    x = np.random.rand(64,64)
    assert rmse(x, x) == 0.0
"""
}

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True) if "/" in path else None
    with open(path, "w") as f:
        f.write(dedent(content).strip() + "\n")

def main():
    for path, content in FILES.items():
        write_file(path, content)
    print("✅ Project scaffold created successfully.")

if __name__ == "__main__":
    main()
