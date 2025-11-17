from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
import random
import copy
import logging

from .model_wrapper import ModelWrapper
from .utils import _get_validation_fitness

logger = logging.getLogger("M2N2_SIMULATOR")


class MergeStrategy(ABC):
    """Abstract base class for all merge strategies."""

    @abstractmethod
    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        """
        Merges the state dictionaries of two parent models.

        Args:
            parent1 (ModelWrapper): The first parent model.
            parent2 (ModelWrapper): The second parent model.
            validation_loader (DataLoader, optional): A DataLoader for a
                validation set, required by some strategies. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: The state dictionary for the new child model.
        """
        pass


class AverageMergeStrategy(MergeStrategy):
    """Merges models by averaging their weights."""

    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        parent1_state_dict = parent1.model.state_dict()
        parent2_state_dict = parent2.model.state_dict()
        child_model_state_dict = copy.deepcopy(parent1_state_dict)

        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] + parent2_state_dict[key]) / 2.0

        return child_model_state_dict




class RandomHalfMergeStrategy(MergeStrategy):
    """
    A fast, heuristic-based merge strategy that creates a hybrid by taking
    half the layers from the weaker parent, then performs a single validation
    check to decide whether to keep the hybrid or the original fitter parent.
    This strategy is stateful and reuses a single model object for all
    evaluations to reduce overhead.
    """
    def __init__(self, model_name: str, device: torch.device, num_classes: int):
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        # Create a single, reusable model wrapper for all fitness evaluations.
        self._reusable_wrapper = ModelWrapper(
            model_name=self.model_name,
            niche_classes=list(range(self.num_classes)),
            device=self.device,
            num_classes=self.num_classes
        )

    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        if validation_loader is None:
            raise ValueError("The 'RandomHalfMergeStrategy' requires a 'validation_loader'.")

        fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        weaker_parent = parent2 if parent1.fitness >= parent2.fitness else parent1
        logger.info(f"  - Using fitter parent (Fitness: {fitter_parent.fitness:.2f}) as base.")

        fitter_state_dict = fitter_parent.model.state_dict()
        weaker_state_dict = weaker_parent.model.state_dict()
        hybrid_state_dict = copy.deepcopy(fitter_state_dict)

        # Get a list of layer prefixes to choose from
        layer_prefixes = sorted(list(set([k.split('.')[0] for k in fitter_state_dict.keys()])))

        # Randomly choose half of the layers to swap from the weaker parent
        num_layers_to_swap = len(layer_prefixes) // 2
        layers_to_swap = random.sample(layer_prefixes, num_layers_to_swap)
        logger.info(f"  - Randomly selected {num_layers_to_swap} layers to swap: {layers_to_swap}")

        # Swap the selected layers
        for key in hybrid_state_dict:
            prefix = key.split('.')[0]
            if prefix in layers_to_swap:
                hybrid_state_dict[key] = weaker_state_dict[key]

        # --- Single Validation Step ---
        # Use a single batch for quick validation to avoid overfitting on the validation set
        try:
            validation_batch = next(iter(validation_loader))
        except StopIteration:
            raise ValueError("Validation loader is empty. Cannot use this merge strategy.")

        # The reusable wrapper's state is managed by `_get_validation_fitness`,
        # which loads the correct state dict before each evaluation. There is
        # no need to save/restore its state here.
        base_fitness = _get_validation_fitness(self._reusable_wrapper, validation_loader, batch=validation_batch, model_state_dict=fitter_state_dict)
        hybrid_fitness = _get_validation_fitness(self._reusable_wrapper, validation_loader, batch=validation_batch, model_state_dict=hybrid_state_dict)

        logger.info(f"  - Fitter Parent Fitness (1 batch): {base_fitness:.2f}%")
        logger.info(f"  - Hybrid Model Fitness (1 batch): {hybrid_fitness:.2f}%")

        # Return the state dict of the better performing model
        if hybrid_fitness > base_fitness:
            logger.info("  - Hybrid model performed better. Keeping the hybrid.")
            return hybrid_state_dict
        else:
            logger.info("  - Fitter parent performed better. Discarding the hybrid.")
            return fitter_state_dict


class FitnessWeightedMergeStrategy(MergeStrategy):
    """Merges models using a fitness-weighted average of their weights."""

    def __init__(self, dampening_factor: float = 25.0):
        self.dampening_factor = dampening_factor

    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        parent1_state_dict = parent1.model.state_dict()
        parent2_state_dict = parent2.model.state_dict()
        child_model_state_dict = copy.deepcopy(parent1_state_dict)

        dampened_fitness1 = parent1.fitness + self.dampening_factor
        dampened_fitness2 = parent2.fitness + self.dampening_factor
        total_dampened_fitness = dampened_fitness1 + dampened_fitness2

        if total_dampened_fitness == 0:
            weight1, weight2 = 0.5, 0.5
        else:
            weight1 = dampened_fitness1 / total_dampened_fitness
            weight2 = dampened_fitness2 / total_dampened_fitness

        logger.info(f"  - Dampened weights: Parent 1 ({weight1:.2f}), Parent 2 ({weight2:.2f})")

        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] * weight1) + (parent2_state_dict[key] * weight2)

        return child_model_state_dict


class LayerWiseMergeStrategy(MergeStrategy):
    """Merges models by randomly selecting entire layers from parents."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        parent1_state_dict = parent1.model.state_dict()
        parent2_state_dict = parent2.model.state_dict()
        child_model_state_dict = copy.deepcopy(parent1_state_dict)

        rng = random.Random(self.seed)
        layer_prefixes = sorted(list(set([k.split('.')[0] for k in parent1_state_dict.keys()])))
        parent_choices = {p: rng.choice([1, 2]) for p in layer_prefixes}

        for key in child_model_state_dict:
            prefix = key.split('.')[0]
            if parent_choices[prefix] == 2:
                child_model_state_dict[key] = parent2_state_dict[key]

        return child_model_state_dict
