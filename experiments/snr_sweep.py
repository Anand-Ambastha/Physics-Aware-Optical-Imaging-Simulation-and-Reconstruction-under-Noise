from optical_imaging.dataset import generate_object
from optical_imaging.psf import airy_psf
from optical_imaging.forward_model import forward_model
from optical_imaging.noise import add_noise
from optical_imaging.reconstruction import wiener_deconvolution
from optical_imaging.metrics import rmse
import matplotlib.pyplot as plt
from optical_imaging.config import IMAGE_SIZE, PHOTON_LEVELS
from optical_imaging.plot_utils import save_figure
obj = generate_object(IMAGE_SIZE)
psf = airy_psf(0.5)

from optical_imaging.metrics import rmse, nrmse

errors = []
norm_errors = []

for photons in PHOTON_LEVELS:
    y = add_noise(forward_model(obj, psf), photons)
    x_hat = wiener_deconvolution(y, psf, K=0.01)

    e = rmse(obj, x_hat)
    ne = nrmse(obj, x_hat)

    errors.append(e)
    norm_errors.append(ne)

    print(
        f"Photons: {photons}, "
        f"RMSE: {e:.4f}, "
        f"NRMSE: {ne:.4f}"
    )

plt.figure()
plt.plot(PHOTON_LEVELS, errors, marker="o", label="RMSE")
plt.plot(PHOTON_LEVELS, norm_errors, marker="s", label="NRMSE")
plt.xlabel("Photon Count")
plt.ylabel("Error")
plt.title("Reconstruction Error vs Photon Count (Fixed Regularization)")
plt.legend()
plt.grid(True)

save_figure("error_vs_photons_fixed_K.png")