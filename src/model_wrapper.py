from __future__ import annotations
from typing import List
import torch
from torch import nn

from .model import CifarCNN, MnistCNN, LLMClassifier, ResNetClassifier
from .enums import ModelName


class ModelWrapper:
    """A wrapper to hold a model and its evolutionary metadata.

    This class encapsulates a model and its associated context, such as its
    specialized niche, its fitness score, and the model architecture.

    Attributes:
        model_name (ModelName): The name of the model architecture.
        niche_classes (list[int]): A list of class indices the model is
            specialized in. An empty or full list implies a generalist.
        device (str): The device ('cpu' or 'cuda') on which the model's
            tensors are allocated.
        model (torch.nn.Module): The underlying neural network model instance.
        fitness (float): The fitness score of the model. Initialized to 0.0.
        fitness_is_current (bool): A flag to indicate if the fitness score
            is up-to-date. Initialized to `False`.
    """
    model_name: ModelName
    niche_classes: List[int]
    device: str
    model: nn.Module
    fitness: float
    fitness_is_current: bool

    def __init__(self, model_name: ModelName, niche_classes: List[int], device: str = 'cpu'):
        """Initializes the ModelWrapper with a model and its niche.

        Args:
            model_name (ModelName): The name of the model to instantiate.
            niche_classes (list[int]): The list of class indices for the
                model's specialized niche.
            device (str, optional): The device to run the model on.
                Defaults to 'cpu'.
        """
        self.model_name = model_name
        self.niche_classes = niche_classes
        self.device = device

        if self.model_name == ModelName.CIFAR10:
            self.model = CifarCNN().to(device)
        elif self.model_name == ModelName.MNIST:
            self.model = MnistCNN().to(device)
        elif self.model_name == ModelName.LLM:
            self.model = LLMClassifier().to(device)
        elif self.model_name == ModelName.RESNET:
            self.model = ResNetClassifier().to(device)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        self.fitness = 0.0
        # This flag prevents redundant evaluations.
        self.fitness_is_current = False
