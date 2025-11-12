"""
Provides utility functions for the M2N2 simulation.

This module contains helper functions that are used across the application,
such as for ensuring experiment reproducibility.
"""
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Sets the seed for all relevant random number generators to ensure
    reproducibility.

    This function sets the seed for Python's `random` module, `numpy`, and
    `torch` (for both CPU and CUDA). It also configures CUDA to use
    deterministic algorithms where available, which is crucial for
    reproducible deep learning experiments.

    Args:
        seed (int): The integer value to use as the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Using deterministic algorithms can impact performance but is key for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
