import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from optical_imaging.dataset import generate_object
from optical_imaging.psf import airy_psf
from optical_imaging.forward_model import forward_model
from optical_imaging.noise import add_noise


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def train_cnn(psf, photons, n_samples=200, epochs=5, lr=1e-3):
    """
    Train CNN on physics-simulated data.
    This is a denoiser, not a super-resolution model.
    """

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for _ in range(epochs):
        for _ in range(n_samples):
            obj = generate_object(64)
            y = add_noise(forward_model(obj, psf), photons)

            inp = torch.tensor(y[None, None], dtype=torch.float32)
            tgt = torch.tensor(obj[None, None], dtype=torch.float32)

            optimizer.zero_grad()
            out = model(inp)
            loss = loss_fn(out, tgt)
            loss.backward()
            optimizer.step()

    return model


def cnn_denoise(model, image):
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(image[None, None], dtype=torch.float32)
        out = model(inp)
    return out.squeeze().numpy()
