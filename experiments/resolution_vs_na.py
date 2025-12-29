import matplotlib.pyplot as plt
from optical_imaging.config import NA_VALUES, WAVELENGTH
from optical_imaging.plot_utils import save_figure
cutoffs = [2 * na / WAVELENGTH for na in NA_VALUES]

plt.figure()
plt.plot(NA_VALUES, cutoffs, marker="o")
plt.xlabel("Numerical Aperture (NA)")
plt.ylabel("Cutoff Frequency (cycles/m)")
plt.title("Diffraction-Limited Resolution vs NA")
plt.grid(True)

save_figure("resolution_vs_na.png")
