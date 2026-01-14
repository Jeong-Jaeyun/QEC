import os
import random

try:
    import numpy as np
except Exception:
    np = None


def set_global_seed(seed: int = 11):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
