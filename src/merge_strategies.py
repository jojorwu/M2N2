"""
Defines the strategies for merging two parent models into a single child.

This module implements the "crossover" phase of the evolutionary algorithm.
Merge strategies define how the weights (parameters) of two parent neural
networks are combined to create a new child network. This allows for various
approaches, from simple averaging to more complex, fitness-aware, or
layer-by-layer constructions.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
import random
import copy
import logging

from .model_wrapper import ModelWrapper
from .model_factory import create_model

logger = logging.getLogger("M2N2_SIMULATOR")

class MergeStrategy(ABC):
    """
    Abstract base class for all model merge strategies.

    This class provides the interface that all merge strategy implementations
    must follow. Subclasses are required to implement the `merge` method, which
    contains the core logic for combining the parent models.
    """
    @abstractmethod
    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        """
        Merges the state dictionaries of two parent models.

        Args:
            parent1 (ModelWrapper): The first parent model.
            parent2 (ModelWrapper): The second parent model.
            validation_loader (DataLoader, optional): A DataLoader for a
                validation set. This is required by some advanced strategies
                (like `SequentialConstructiveMergeStrategy`) to evaluate the
                quality of the merge. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: The state dictionary for the new child model.
        """
        pass

class AverageMergeStrategy(MergeStrategy):
    """
    Merges models by taking the simple arithmetic average of their weights.

    This is the most straightforward merging technique. For each parameter, the
    value in the child model is the average of the values from the two parents.
    """
    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        parent1_state_dict = parent1.model.state_dict()
        parent2_state_dict = parent2.model.state_dict()
        child_model_state_dict = copy.deepcopy(parent1_state_dict)

        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] + parent2_state_dict[key]) / 2.0
        return child_model_state_dict

class FitnessWeightedMergeStrategy(MergeStrategy):
    """
    Merges models using a weighted average based on the parents' fitness.

    This strategy gives more influence to the "fitter" parent. The weights for
    the average are calculated by applying a softmax function to the fitness
    scores of the two parents. This ensures that the weights are positive, sum
    to 1, and proportionally reflect the parents' performance.
    """
    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        parent1_state_dict = parent1.model.state_dict()
        parent2_state_dict = parent2.model.state_dict()
        child_model_state_dict = copy.deepcopy(parent1_state_dict)

        fitness_tensor = torch.tensor([parent1.fitness, parent2.fitness], dtype=torch.float32)
        weights = torch.nn.functional.softmax(fitness_tensor, dim=0)
        weight1, weight2 = weights[0].item(), weights[1].item()

        logger.info(f"  - Softmax weights: Parent 1 ({weight1:.2f}), Parent 2 ({weight2:.2f})")

        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] * weight1) + (parent2_state_dict[key] * weight2)
        return child_model_state_dict

class LayerWiseMergeStrategy(MergeStrategy):
    """
    Merges models by randomly selecting entire layers from either parent.

    For each layer (or block of layers) in the model architecture, this strategy
    makes a random choice to copy the entire layer's weights from either
    `parent1` or `parent2`. This can be useful for combining functional blocks
    from different specialist models.
    """
    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the LayerWiseMergeStrategy.

        Args:
            seed (Optional[int], optional): A random seed to ensure
                reproducible layer selection. Defaults to None.
        """
        self.seed = seed

    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        parent1_state_dict = parent1.model.state_dict()
        parent2_state_dict = parent2.model.state_dict()
        child_model_state_dict = copy.deepcopy(parent1_state_dict)

        rng = random.Random(self.seed)
        layer_prefixes = sorted(list(set([k.split('.')[0] for k in parent1_state_dict.keys()])))

        for prefix in layer_prefixes:
            if rng.choice([True, False]):
                # Take this layer from parent 2
                for key in parent2_state_dict:
                    if key.startswith(prefix):
                        child_model_state_dict[key] = parent2_state_dict[key]
        return child_model_state_dict

class SequentialConstructiveMergeStrategy(MergeStrategy):
    """
    Merges models by intelligently building a child layer by layer.

    This advanced strategy starts with the "fitter" parent as a base and
    iteratively tests swapping in each layer from the "weaker" parent. A swap
    is only kept if it results in an improvement in fitness, which is measured
    on a small validation batch to keep the process efficient.
    """
    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        if not validation_loader:
            raise ValueError("The 'sequential_constructive' strategy requires a 'validation_loader'.")

        fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        weaker_parent = parent2 if parent1.fitness >= parent2.fitness else parent1
        logger.info(f"  - Base parent (fitter): Fitness {fitter_parent.fitness:.2f}")

        temp_model = create_model(fitter_parent.model_name, fitter_parent.model.num_classes, fitter_parent.device)
        temp_model.load_state_dict(copy.deepcopy(fitter_parent.model.state_dict()))
        temp_model_wrapper = ModelWrapper(
            model_name=fitter_parent.model_name,
            model=temp_model,
            niche_classes=list(range(fitter_parent.model.num_classes)),
            device=fitter_parent.device
        )

        try:
            validation_batch = next(iter(validation_loader))
        except StopIteration:
            raise ValueError("Validation loader is empty, cannot use this strategy.")

        best_fitness = temp_model_wrapper._calculate_accuracy(batch=validation_batch)
        logger.info(f"  - Initial child validation fitness (on one batch): {best_fitness:.2f}%")

        layer_prefixes = sorted(list(set([k.split('.')[0] for k in fitter_parent.model.state_dict().keys()])))

        for prefix in layer_prefixes:
            original_layers = {k: v.clone() for k, v in temp_model.state_dict().items() if k.startswith(prefix)}

            # Swap in the layer from the weaker parent
            for key in original_layers:
                if key in weaker_parent.model.state_dict():
                    temp_model.state_dict()[key].copy_(weaker_parent.model.state_dict()[key])

            current_fitness = temp_model_wrapper._calculate_accuracy(batch=validation_batch)

            if current_fitness >= best_fitness:
                logger.info(f"  - Swapping layer '{prefix}' improved fitness to {current_fitness:.2f}%. Keeping.")
                best_fitness = current_fitness
            else:
                logger.info(f"  - Swapping layer '{prefix}' did not improve fitness. Reverting.")
                for key, original_tensor in original_layers.items():
                    temp_model.state_dict()[key].copy_(original_tensor)

        return temp_model.state_dict()
