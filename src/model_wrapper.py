from __future__ import annotations
from typing import List, Optional, Any, Tuple
import io
import torch
import logging
from torch import nn

from .model import CifarCNN, LLMClassifier, ResNetClassifier
import os
import re
from .model_factory import create_model
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

    def __init__(self, model_name: ModelName, model: nn.Module, niche_classes: List[int], device: str = 'cpu'):
        """Initializes the ModelWrapper with a model and its niche.
        Args:
            model_name (ModelName): The name of the model architecture.
            model (nn.Module): A pre-instantiated model object.
            niche_classes (list[int]): The list of class indices for the
                model's specialized niche.
            device (str, optional): The device to run the model on.
                Defaults to 'cpu'.
        """
        self.model_name = model_name
        self.model = model
        self.niche_classes = niche_classes
        self.device = device
        self.fitness = 0.0
        # This flag prevents redundant evaluations.
        self.fitness_is_current = False

    @staticmethod
    def from_file(filepath: str, model_name: ModelName, num_classes: int, device: str = 'cpu') -> Optional[ModelWrapper]:
        """
        Creates a ModelWrapper instance by loading a model from a file.

        This factory method encapsulates the logic for parsing the model's
        metadata (niche, fitness) from the filename, creating the model
        architecture, and loading the saved state dictionary.

        Args:
            filepath (str): The path to the saved model file (.pth).
            model_name (ModelName): The name of the model architecture.
            num_classes (int): The number of output classes for the model.
            device (str, optional): The device to load the model onto.
                Defaults to 'cpu'.

        Returns:
            ModelWrapper | None: An initialized ModelWrapper instance if the
            filename is parsed successfully, otherwise None.
        """
        match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(filepath))
        if match:
            niche_classes = [int(n) for n in match.group(1).split('_')]
            fitness = float(match.group(2))
            model = create_model(
                model_name=model_name,
                num_classes=num_classes,
                device=device
            )
            model.load_state_dict(torch.load(filepath, map_location=device))

            wrapper = ModelWrapper(
                model_name=model_name,
                model=model,
                niche_classes=niche_classes,
                device=device
            )
            wrapper.fitness = fitness
            wrapper.fitness_is_current = False  # Fitness from filename might be stale
            return wrapper
        return None

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

    def _process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to process a batch and return predictions and targets."""
        if self.model_name == 'LLM':
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target = batch['labels'].to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:  # Handles CIFAR10, MNIST, etc.
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            if next(self.model.parameters()).dtype == torch.float64:
                data = data.double()
            output = self.model(data)

        _, predicted = torch.max(output.data, 1)
        return predicted, target

    def _calculate_accuracy(self, data_loader: Optional[DataLoader] = None, batch: Optional[Any] = None) -> float:
        """A generic helper to calculate accuracy.
        This function can calculate accuracy on either a full `DataLoader` or
        a single, pre-fetched `batch`. This is a key optimization for
        strategies like sequential constructive merging, as it allows for
        rapid evaluation on a consistent batch without repeated data loading.
        Args:
            data_loader (DataLoader, optional): A DataLoader to evaluate.
                Defaults to None.
            batch (Any, optional): A single pre-fetched batch to evaluate.
                Defaults to None.
        Returns:
            float: The calculated accuracy as a percentage.
        """
        self.model.eval()
        correct = 0
        total = 0

        if batch is None and data_loader is None:
            raise ValueError("Either data_loader or batch must be provided.")

        with torch.no_grad():
            data_source = [batch] if batch is not None else data_loader
            for b in data_source:
                predicted, target = self._process_batch(b)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100 * correct / total if total > 0 else 0.0

    def evaluate_by_class(self, dataset_name: str, subset_percentage: float = 1.0, seed: Optional[int] = None) -> List[float]:
        """Evaluates a model's accuracy on each individual class.

        This function is used to identify a model's strengths and weaknesses,
        which is crucial for the advanced mate selection strategy. It does not
        modify the model wrapper.

        Args:
            dataset_name (str): The name of the dataset to use for evaluation.
            subset_percentage (float, optional): The fraction of the test set to use for evaluation. Defaults to 1.0.
            seed (int, optional): A seed for the random number generator to
                ensure deterministic data splitting. Defaults to None.

        Returns:
            list[float]: A list of accuracy percentages, where the index of the
                list corresponds to the class index.
        """
        # We always evaluate on the full test set to measure general performance
        _, _, test_loader, _ = get_dataloaders(dataset_name=dataset_name, model_name=self.model_name, subset_percentage=subset_percentage, validation_split=0, seed=seed) # No validation split needed here
        self.model.eval()

        num_classes = self.model.num_classes
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        with torch.no_grad():
            for batch in test_loader:
                predicted, target = self._process_batch(batch)
                c = (predicted == target).squeeze()

                for i in range(len(target)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)

        return class_accuracies

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
