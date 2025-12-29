import os
import matplotlib.pyplot as plt

def save_figure(filename, dpi=300):
    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", filename)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
