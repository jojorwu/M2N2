from enum import Enum

class ModelName(str, Enum):
    CIFAR10 = 'CIFAR10'
    MNIST = 'MNIST'
    LLM = 'LLM'
    RESNET = 'RESNET'

class DatasetName(str, Enum):
    CIFAR10 = 'CIFAR10'
    MNIST = 'MNIST'
    LLM = 'LLM'
