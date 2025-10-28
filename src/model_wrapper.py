from __future__ import annotations
from typing import List, Optional
import io
import torch
import logging
from torch import nn

from .model import CifarCNN, LLMClassifier, ResNetClassifier
from .enums import ModelName
from .data import get_dataloaders

logger = logging.getLogger("M2N2_SIMULATOR")

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

    def evaluate(self, dataset_name: str, subset_percentage: float = 1.0, seed: Optional[int] = None) -> float:
        """Evaluates fitness on the full test set and updates the wrapper.

        This function skips evaluation if the model's fitness is already
        marked as current. Otherwise, it calculates the accuracy on the test
        set and updates the `fitness` and `fitness_is_current` attributes.

        Args:
            dataset_name (str): The name of the dataset to use for evaluation.
            subset_percentage (float, optional): The fraction of the test set to use for evaluation. Defaults to 1.0.
            seed (int, optional): A seed for the random number generator to
                ensure deterministic data splitting. Defaults to None.

        Returns:
            float: The calculated accuracy (fitness) of the model as a percentage.
        """
        if self.fitness_is_current:
            logger.debug(f"  - Skipping evaluation for model with up-to-date fitness: {self.fitness:.2f}%")
            return self.fitness

        _, _, test_loader, _ = get_dataloaders(dataset_name=dataset_name, model_name=self.model_name, subset_percentage=subset_percentage, validation_split=0, seed=seed) # No validation split needed here
        accuracy = self._calculate_accuracy(test_loader)
        self.fitness = accuracy
        self.fitness_is_current = True
        return accuracy

    def _calculate_accuracy(self, data_loader, batch=None) -> float:
        """
        A generic helper to calculate accuracy on a given data loader or a single
        batch.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            data_source = [batch] if batch else data_loader
            for b in data_source:
                if self.model_name == 'LLM':
                    input_ids = b['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                else:
                    data, target = b
                    data, target = data.to(self.device), target.to(self.device)
                    if next(self.model.parameters()).dtype == torch.float64:
                        data = data.double()
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
        return 100 * correct / total if total > 0 else 0.0

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

        # Check for basic attribute equality
        if self.model_name != other.model_name or self.niche_classes != other.niche_classes:
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

        return hash((self.model_name, tuple(self.niche_classes), state_dict_bytes))
