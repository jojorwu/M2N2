"""
Defines enumerations used throughout the M2N2 simulation.

Using enums instead of raw strings for categorical parameters (like model or
dataset names) improves code clarity, enforces type safety, and prevents
common errors caused by typos.
"""
from enum import Enum

class ModelName(str, Enum):
    """Enumeration for the supported model architectures."""
    CNN = 'CNN'
    LLM = 'LLM'
    RESNET = 'RESNET'

class DatasetName(str, Enum):
    """Enumeration for the supported datasets."""
    CIFAR10 = 'CIFAR10'
    MNIST = 'MNIST'
    LLM = 'LLM'
