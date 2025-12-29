"""
Run all experiments for the physics-aware optical imaging study.

This script:
- executes all experiment modules
- saves all figures to ./figures
- prints key numerical results
- is safe for headless / batch execution

Usage:
    uv run python experiments/run_all_experiments.py
"""

import sys
import os

# Ensure project root is on path (safe when using editable install)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("=" * 60)
print("Running Physics-Aware Optical Imaging Experiments")
print("=" * 60)

# ---------------------------------------------------------------------
# 1. Resolution vs NA (theoretical optics)
# ---------------------------------------------------------------------
print("\n[1/3] Running resolution_vs_na.py")
try:
    import experiments.resolution_vs_na  # noqa: F401
    print("✔ Resolution vs NA completed")
except Exception as e:
    print("✘ Resolution vs NA failed")
    raise e

# ---------------------------------------------------------------------
# 2. Failure mode visualization (inverse instability)
# ---------------------------------------------------------------------
print("\n[2/3] Running failure_modes.py")
try:
    import experiments.failure_modes  # noqa: F401
    print("✔ Failure mode visualization completed")
except Exception as e:
    print("✘ Failure mode visualization failed")
    raise e

# ---------------------------------------------------------------------
# 3. RMSE vs photon count (fixed regularization)
# ---------------------------------------------------------------------
print("\n[3/3] Running snr_sweep.py")
try:
    import experiments.snr_sweep  # noqa: F401
    print("✔ SNR sweep completed")
except Exception as e:
    print("✘ SNR sweep failed")
    raise e

print("\nAll experiments completed successfully.")
print("Figures saved to ./figures/")
print("=" * 60)
