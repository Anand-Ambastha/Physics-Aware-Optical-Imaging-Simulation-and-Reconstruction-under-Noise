import matplotlib.pyplot as plt
from optical_imaging.dataset import generate_object
from optical_imaging.psf import airy_psf
from optical_imaging.forward_model import forward_model
from optical_imaging.noise import add_noise
from optical_imaging.reconstruction import wiener_deconvolution
from optical_imaging.plot_utils import save_figure
obj = generate_object(256)
psf = airy_psf(0.7)
y = add_noise(forward_model(obj, psf), photons=20)
x_hat = wiener_deconvolution(y, psf, K=0.001)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Ground Truth"); plt.imshow(obj); plt.axis("off")
plt.subplot(1,3,2); plt.title("Measurement (Low photons)"); plt.imshow(y); plt.axis("off")
plt.subplot(1,3,3); plt.title("Wiener Reconstruction"); plt.imshow(x_hat); plt.axis("off")

save_figure("failure_mode_low_photons.png")