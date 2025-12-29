import matplotlib.pyplot as plt
from optical_imaging.dataset import generate_object
from optical_imaging.psf import airy_psf
from optical_imaging.forward_model import forward_model
from optical_imaging.noise import add_noise
from optical_imaging.reconstruction import wiener_deconvolution
from optical_imaging.cnn import SimpleCNN, train_cnn, cnn_denoise
from optical_imaging.metrics import rmse
from optical_imaging.plot_utils import save_figure

obj = generate_object(128)
psf = airy_psf(0.5)

y = add_noise(forward_model(obj, psf), photons=100)

# Classical
x_wiener = wiener_deconvolution(y, psf, K=0.01)

# CNN (trained on simulated data)
print("Model Training Started")
model = train_cnn(psf, photons=100)
x_cnn = cnn_denoise(model, y)

print("RMSE Wiener:", rmse(obj, x_wiener))
print("RMSE CNN:", rmse(obj, x_cnn))

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Ground Truth"); plt.imshow(obj); plt.axis("off")
plt.subplot(1,3,2); plt.title("Wiener"); plt.imshow(x_wiener); plt.axis("off")
plt.subplot(1,3,3); plt.title("CNN"); plt.imshow(x_cnn); plt.axis("off")

save_figure("cnn_vs_wiener.png")
