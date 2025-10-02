import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Sets the seed for random number generators in random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The following two lines are for ensuring reproducible results on CUDA.
        # They can have a performance impact, so they are often disabled.
        # For this project, reproducibility is more important.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
