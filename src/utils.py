import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The following two lines are often recommended for deterministic GPU behavior,
        # but they can impact performance. They are included here for completeness.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False