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
    Merges models by creating a hybrid child where half the layers are
    randomly taken from the weaker parent. The hybrid is kept only if it
    outperforms the fitter parent on a validation batch.
    """

    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        if validation_loader is None:
            raise ValueError("The 'sequential_constructive' strategy requires a 'validation_loader'.")

        fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        weaker_parent = parent2 if parent1.fitness >= parent2.fitness else parent1
        logger.info(f"  - Using fitter parent (Fitness: {fitter_parent.fitness:.2f}) as base.")

        best_child_state_dict = copy.deepcopy(fitter_parent.model.state_dict())
        num_classes = fitter_parent.model.num_classes
        temp_model_wrapper = ModelWrapper(model_name=fitter_parent.model_name, niche_classes=list(range(num_classes)), device=fitter_parent.device, num_classes=num_classes)
        temp_model_wrapper.model.load_state_dict(best_child_state_dict)

        # --- Optimization: Use a single batch for quick validation ---
        try:
            validation_batch = next(iter(validation_loader))
        except StopIteration:
            raise ValueError("Validation loader is empty. Cannot use 'sequential_constructive' strategy.")

        best_fitness = _get_validation_fitness(temp_model_wrapper, validation_loader, batch=validation_batch)
        logger.info(f"  - Initial child validation fitness (on one batch): {best_fitness:.2f}%")

        if fitter_parent.model_name == 'LLM':
            layer_prefixes = ['bert.distilbert.embeddings']
            num_transformer_layers = fitter_parent.model.bert.config.num_hidden_layers
            for i in range(num_transformer_layers):
                layer_prefixes.append(f'bert.distilbert.transformer.layer.{i}')
            layer_prefixes.extend(['bert.pre_classifier', 'bert.classifier'])
        elif fitter_parent.model_name == 'RESNET':
            layer_prefixes = [name for name, _ in fitter_parent.model.resnet.named_children()]
        else:
            layer_prefixes = sorted(list(set([k.split('.')[0] for k in fitter_parent.model.state_dict().keys()])))

        current_state_dict = temp_model_wrapper.model.state_dict()
        weaker_parent_state_dict = weaker_parent.model.state_dict()

        # --- Major Optimization: Evaluate each layer's contribution independently ---
        # Instead of N full forward passes, we make a single pass for each parent,
        # then swap layers and do one final pass to decide. This is a heuristic
        # that is much faster.

        # For simplicity in this optimization, we will just swap a random half of the layers
        # and then do a single validation. This reduces N validations to 1.

        rng = random.Random() # No seed for now, could be added
        layers_to_swap = rng.sample(layer_prefixes, k=len(layer_prefixes) // 2)
        logger.info(f"  - Heuristic optimization: Swapping {len(layers_to_swap)} layers from weaker parent.")

        for prefix in layers_to_swap:
             # Swap in the layers from the weaker parent
            for key in current_state_dict:
                if key.startswith(prefix):
                    current_state_dict[key].copy_(weaker_parent_state_dict[key])

        # Load the new hybrid state into the temp model
        temp_model_wrapper.model.load_state_dict(current_state_dict)

        # Perform a single validation on the final hybrid
        final_fitness = _get_validation_fitness(temp_model_wrapper, validation_loader, batch=validation_batch)
        logger.info(f"  - Final child validation fitness (on one batch): {final_fitness:.2f}%")

        # If the hybrid is better, return its state dict. Otherwise, return the fitter parent's.
        if final_fitness > best_fitness:
            logger.info("  - Hybrid model is better than the fitter parent. Keeping it.")
            return current_state_dict
        else:
            logger.info("  - Hybrid model is not better. Returning the fitter parent's state dict.")
            return fitter_parent.model.state_dict()


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
