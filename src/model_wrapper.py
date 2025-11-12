"""
Provides a wrapper class for PyTorch models to manage evolutionary metadata.

This module defines the `ModelWrapper` class, which is a cornerstone of the
simulation. It encapsulates a `torch.nn.Module` and attaches essential
metadata for the evolutionary process, such as its fitness score, its
specialized data "niche," and its architectural details. This allows models
to be treated as individuals in a population that can be evaluated, selected,
and modified.
"""
from __future__ import annotations
from typing import List, Optional, Any, Tuple
import io
import torch
import logging
from torch import nn
import os
import re
from .model_factory import create_model
from .enums import ModelName
from .data import get_dataloaders

logger = logging.getLogger("M2N2_SIMULATOR")

class ModelWrapper:
    """
    A wrapper to hold a model and its evolutionary metadata.

    This class encapsulates a model and its associated context, such as its
    specialized niche, its fitness score, and the model architecture. It
    provides methods for evaluation, saving, and loading, and implements
    equality and hashing for use in collections.

    Attributes:
        model_name (ModelName): The name of the model architecture.
        niche_classes (List[int]): A list of class indices the model is
            specialized in. An empty or full list implies a generalist.
        device (str): The device ('cpu' or 'cuda') on which the model's
            tensors are allocated.
        model (nn.Module): The underlying neural network model instance.
        fitness (float): The fitness score of the model, typically accuracy.
        fitness_is_current (bool): A flag to indicate if the fitness score
            is up-to-date, preventing redundant evaluations.
    """
    def __init__(self, model_name: ModelName, model: nn.Module, niche_classes: List[int], device: str = 'cpu'):
        """
        Initializes the ModelWrapper.

        Args:
            model_name (ModelName): The name of the model architecture.
            model (nn.Module): A pre-instantiated model object.
            niche_classes (List[int]): The list of class indices for the
                model's specialized niche.
            device (str, optional): The device to run the model on.
                Defaults to 'cpu'.
        """
        self.model_name = model_name
        self.model = model
        self.niche_classes = niche_classes
        self.device = device
        self.fitness = 0.0
        self.fitness_is_current = False

    @staticmethod
    def from_file(filepath: str, model_name: ModelName, num_classes: int, device: str = 'cpu') -> Optional[ModelWrapper]:
        """
        Creates a ModelWrapper by loading a model from a file.

        This factory method parses metadata (niche, fitness) from the filename,
        creates the model architecture, and loads the saved state dictionary.

        Args:
            filepath (str): The path to the saved model file (.pth).
            model_name (ModelName): The name of the model architecture.
            num_classes (int): The number of output classes for the model.
            device (str, optional): The device to load the model onto.
                Defaults to 'cpu'.

        Returns:
            Optional[ModelWrapper]: An initialized ModelWrapper, or None if
            the filename cannot be parsed or the file fails to load.
        """
        try:
            match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(filepath))
            if not match:
                logger.warning(f"Could not parse niche/fitness from filename: {filepath}")
                return None

            niche_classes = [int(n) for n in match.group(1).split('_')]
            fitness = float(match.group(2))
            model = create_model(model_name, num_classes, device)
            model.load_state_dict(torch.load(filepath, map_location=device))

            wrapper = ModelWrapper(model_name, model, niche_classes, device)
            wrapper.fitness = fitness
            wrapper.fitness_is_current = True # Assume saved fitness is current
            return wrapper
        except Exception as e:
            logger.warning(f"Failed to load model from {filepath}: {e}")
            return None

    def save(self, filepath: str) -> None:
        """
        Saves the model's state dictionary to the specified file.

        Args:
            filepath (str): The path to save the model file to.
        """
        torch.save(self.model.state_dict(), filepath)

    def evaluate(self, dataset_name: ModelName, subset_percentage: float = 1.0, seed: Optional[int] = None) -> float:
        """
        Evaluates fitness on the full test set and updates the wrapper.

        Skips evaluation if fitness is current. Otherwise, calculates test set
        accuracy and updates the `fitness` and `fitness_is_current` attributes.

        Args:
            dataset_name (ModelName): The dataset to use for evaluation.
            subset_percentage (float, optional): Fraction of the test set
                to use. Defaults to 1.0.
            seed (Optional[int], optional): Seed for reproducible data splitting.
                Defaults to None.

        Returns:
            float: The calculated accuracy (fitness) as a percentage.
        """
        if self.fitness_is_current:
            return self.fitness

        _, _, test_loader, _ = get_dataloaders(
            dataset_name, self.model_name, subset_percentage=subset_percentage,
            validation_split=0, seed=seed
        )
        self.fitness = self._calculate_accuracy(data_loader=test_loader)
        self.fitness_is_current = True
        return self.fitness

    def _process_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a data batch and returns model outputs and targets.

        This helper centralizes logic for handling different data formats
        (e.g., CNN inputs vs. LLM inputs) and moves data to the correct device.

        Args:
            batch (Any): A single data batch from a DataLoader.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the raw
                model output and the ground truth target tensor.
        """
        if self.model_name == ModelName.LLM:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target = batch['labels'].to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            if next(self.model.parameters()).dtype == torch.float64:
                data = data.double()
            output = self.model(data)
        return output, target

    def _calculate_accuracy(self, data_loader: Optional[DataLoader] = None, batch: Optional[Any] = None) -> float:
        """
        Calculates accuracy on a full `DataLoader` or a single `batch`.

        This optimization allows for rapid evaluation on a consistent batch
        during certain merge strategies without repeated data loading.

        Args:
            data_loader (Optional[DataLoader], optional): A DataLoader to
                evaluate. Defaults to None.
            batch (Optional[Any], optional): A single pre-fetched batch to
                evaluate. Defaults to None.

        Returns:
            float: The calculated accuracy as a percentage.

        Raises:
            ValueError: If neither `data_loader` nor `batch` is provided.
        """
        if not data_loader and not batch:
            raise ValueError("Either data_loader or batch must be provided.")

        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            data_source = [batch] if batch else data_loader
            for b in data_source:
                output, target = self._process_batch(b)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total if total > 0 else 0.0

    def evaluate_by_class(self, dataset_name: ModelName, subset_percentage: float = 1.0, seed: Optional[int] = None) -> List[float]:
        """
        Evaluates the model's accuracy on each individual class.

        Used to identify a model's strengths and weaknesses, which is crucial
        for mate selection.

        Args:
            dataset_name (ModelName): The dataset to use for evaluation.
            subset_percentage (float, optional): Fraction of the test set
                to use. Defaults to 1.0.
            seed (Optional[int], optional): Seed for reproducible data splitting.
                Defaults to None.

        Returns:
            List[float]: A list of accuracy percentages, where the index of the
                list corresponds to the class index.
        """
        _, _, test_loader, num_classes = get_dataloaders(
            dataset_name, self.model_name, subset_percentage=subset_percentage,
            validation_split=0, seed=seed
        )
        self.model.eval()
        class_correct = list(0. for _ in range(num_classes))
        class_total = list(0. for _ in range(num_classes))
        with torch.no_grad():
            for batch in test_loader:
                output, target = self._process_batch(batch)
                _, predicted = torch.max(output.data, 1)
                for i in range(len(target)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        return [100 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]

    def __eq__(self, other: object) -> bool:
        """
        Checks for equality between two ModelWrapper instances.

        Two wrappers are equal if they have the same model name and identical
        model state dictionaries. Niche classes are intentionally excluded,
        as a merged generalist may be genetically identical to a parent.

        Args:
            other (object): The object to compare against.

        Returns:
            bool: True if the instances are equal, False otherwise.
        """
        if not isinstance(other, ModelWrapper):
            return NotImplemented
        if self.model_name != other.model_name:
            return False

        self_state = self.model.state_dict()
        other_state = other.model.state_dict()
        if self_state.keys() != other_state.keys():
            return False
        return all(torch.equal(self_state[k], other_state[k]) for k in self_state)

    def __hash__(self) -> int:
        """
        Computes a hash for the ModelWrapper instance for use in sets.

        The hash is based on the model name and its state dictionary, allowing
        for the detection of genetically identical models. This is optimized
        by hashing the byte representation of the state dictionary.

        Returns:
            int: The computed hash value.
        """
        with io.BytesIO() as buffer:
            torch.save(self.model.state_dict(), buffer)
            state_dict_bytes = buffer.getvalue()
        return hash((self.model_name, state_dict_bytes))
