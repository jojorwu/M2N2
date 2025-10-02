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




class SequentialConstructiveMergeStrategy(MergeStrategy):
    """
    Merges models by intelligently building a child layer by layer,
    keeping changes only if they improve validation fitness.
    """

    def merge(self, parent1: ModelWrapper, parent2: ModelWrapper, validation_loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        if validation_loader is None:
            raise ValueError("The 'sequential_constructive' strategy requires a 'validation_loader'.")

        fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        weaker_parent = parent2 if parent1.fitness >= parent2.fitness else parent1
        logger.info(f"  - Using fitter parent (Fitness: {fitter_parent.fitness:.2f}) as base.")

        best_child_state_dict = copy.deepcopy(fitter_parent.model.state_dict())
        num_classes = fitter_parent.model.num_classes
        temp_model_wrapper = ModelWrapper(model_name=fitter_parent.model_name, niche_classes=list(range(num_classes)), device=fitter_parent.device)
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
        for prefix in layer_prefixes:
            if fitter_parent.model_name == 'RESNET':
                module_to_check = dict(fitter_parent.model.resnet.named_children()).get(prefix)
                if module_to_check and not list(module_to_check.parameters()):
                    logger.info(f"  - Skipping layer '{prefix}' as it has no learnable parameters.")
                    continue

            # --- Optimization: Avoid deepcopying the entire state dict ---
            # 1. Store the original layers from the best model
            original_layers = {key: current_state_dict[key].clone() for key in current_state_dict if key.startswith(prefix)}

            # 2. Swap in the layers from the weaker parent
            for key in original_layers:
                current_state_dict[key].copy_(weaker_parent.model.state_dict()[key])

            # 3. Evaluate the new configuration
            current_fitness = _get_validation_fitness(temp_model_wrapper, validation_loader, batch=validation_batch)

            # 4. Decide whether to keep or revert the change
            if current_fitness > best_fitness:
                logger.info(f"  - Swapping layer '{prefix}' improved validation fitness to {current_fitness:.2f}%. Keeping it.")
                best_fitness = current_fitness
                # The change is already in current_state_dict, so we just update best_child_state_dict
                for key in original_layers:
                    best_child_state_dict[key].copy_(current_state_dict[key])
            else:
                logger.info(f"  - Swapping layer '{prefix}' did not improve validation fitness ({current_fitness:.2f}%). Reverting.")
                # Revert the change by copying the original layers back
                for key in original_layers:
                    current_state_dict[key].copy_(original_layers[key])

        return best_child_state_dict


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
