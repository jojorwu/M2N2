from __future__ import annotations
from typing import List
import io
import torch
from torch import nn

from .model import CifarCNN, LLMClassifier, ResNetClassifier
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

    def __init__(self, model_name: ModelName, niche_classes: List[int], device: str = 'cpu', num_classes: int = 10):
        """Initializes the ModelWrapper with a model and its niche.

        Args:
            model_name (ModelName): The name of the model to instantiate.
            niche_classes (list[int]): The list of class indices for the
                model's specialized niche.
            device (str, optional): The device to run the model on.
                Defaults to 'cpu'.
            num_classes (int, optional): The number of output classes for the
                model. Defaults to 10.
        """
        self.model_name = model_name
        self.niche_classes = niche_classes
        self.device = device

        if self.model_name == ModelName.CIFAR10:
            self.model = CifarCNN(num_classes=num_classes).to(device)
        elif self.model_name == ModelName.LLM:
            self.model = LLMClassifier(num_labels=num_classes).to(device)
        elif self.model_name == ModelName.RESNET:
            self.model = ResNetClassifier(num_classes=num_classes).to(device)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        self.fitness = 0.0
        # This flag prevents redundant evaluations.
        self.fitness_is_current = False

    def __eq__(self, other: object) -> bool:
        """Checks for equality between two ModelWrapper instances.

        Two wrappers are considered equal if they have the same model name,
        the same niche, and identical model state dictionaries.

        Args:
            other (object): The object to compare against.

        Returns:
            bool: True if the instances are equal, False otherwise.
        """
        if not isinstance(other, ModelWrapper):
            return NotImplemented

        # Check for basic attribute equality. The niche comparison is sorted
        # to ensure that the order of classes does not affect equality.
        if self.model_name != other.model_name or sorted(self.niche_classes) != sorted(other.niche_classes):
            return False

        # Check for model state dictionary equality
        self_state_dict = self.model.state_dict()
        other_state_dict = other.model.state_dict()

        if self_state_dict.keys() != other_state_dict.keys():
            return False

        for key in self_state_dict:
            if not torch.equal(self_state_dict[key], other_state_dict[key]):
                return False

        return True

    def __hash__(self) -> int:
        """Computes a hash for the ModelWrapper instance.

        The hash is based on the model name, its niche, and the model's
        state dictionary. This allows ModelWrapper instances to be used in
        hash-based collections like sets.

        This implementation is optimized for performance by serializing the
        state dictionary to a byte stream and hashing the bytes, which is
        significantly faster than converting tensors to tuples.

        Returns:
            int: The computed hash value.
        """
        # Using an in-memory binary buffer
        with io.BytesIO() as buffer:
            torch.save(self.model.state_dict(), buffer)
            buffer.seek(0)
            state_dict_bytes = buffer.read()

        # The niche_classes are converted to a sorted tuple to ensure the hash
        # is order-independent, matching the behavior of the __eq__ method.
        return hash((self.model_name, tuple(sorted(self.niche_classes)), state_dict_bytes))
