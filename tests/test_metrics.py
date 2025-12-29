import numpy as np
from optical_imaging.metrics import rmse

def test_rmse_zero():
    x = np.random.rand(64,64)
    assert rmse(x, x) == 0.0
