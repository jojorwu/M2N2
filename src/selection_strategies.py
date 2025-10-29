"""
Implements various mate selection strategies for the evolutionary algorithm.

This module uses the Strategy design pattern to decouple the main
evolutionary algorithm from the specific implementation of how parents are
selected for mating. Each strategy is a class that implements a common
interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import random
import logging

from .model_wrapper import ModelWrapper

logger = logging.getLogger("M2N2_SIMULATOR")

class MateSelectionStrategy(ABC):
    """Abstract base class for all mate selection strategies."""
    @abstractmethod
    def select_mates(
        self,
        population: List[ModelWrapper],
        dataset_name: "DatasetName",
        subset_percentage: float = 1.0,
        seed: Optional[int] = None
    ) -> Tuple[Optional[ModelWrapper], Optional[ModelWrapper]]:
        """Selects a pair of parents from the population."""
        pass

class HealingMateSelectionStrategy(MateSelectionStrategy):
    """
    Selects parents using a "healing" strategy.

    This strategy pairs the strongest overall model with a specialist that is
    an expert in the strongest model's weakest area.
    """
    def select_mates(
        self,
        population: List[ModelWrapper],
        dataset_name: "DatasetName",
        subset_percentage: float = 1.0,
        seed: Optional[int] = None
    ) -> Tuple[Optional[ModelWrapper], Optional[ModelWrapper]]:
        logger.info("Selecting mates with healing strategy...")
        if not population:
            return None, None

        parent1 = max(population, key=lambda m: m.fitness)
        logger.info(f"  - Parent 1 is the population's best model (Fitness: {parent1.fitness:.2f}%)")

        logger.info("  - Analyzing Parent 1's performance by class...")
        class_accuracies = parent1.evaluate_by_class(dataset_name=dataset_name, subset_percentage=subset_percentage, seed=seed)
        # Get the top 3 weakest classes to search for a specialist mate
        sorted_class_indices = sorted(range(len(class_accuracies)), key=lambda k: class_accuracies[k])
        top_n_weakest_indices = sorted_class_indices[:3]
        logger.info(f"  - Parent 1's top 3 weakest classes are {top_n_weakest_indices} with accuracies {[f'{class_accuracies[i]:.2f}%' for i in top_n_weakest_indices]}")

        parent2 = None
        for class_index in top_n_weakest_indices:
            logger.info(f"  - Searching for a specialist in Parent 1's weak class: {class_index}")
            specialist_candidates = [
                m for m in population if m.niche_classes == [class_index] and m != parent1
            ]

            if specialist_candidates:
                parent2 = max(specialist_candidates, key=lambda m: m.fitness)
                logger.info(f"  - Found best specialist for class {class_index} as Parent 2 (Fitness: {parent2.fitness:.2f}%)")
                break  # Found a suitable specialist, no need to search further.
        else:
            logger.info("  - No suitable specialist found. Using second-best model as fallback Parent 2.")
            sorted_population = sorted(population, key=lambda m: m.fitness, reverse=True)
            parent2 = next((model for model in sorted_population if model is not parent1 and model != parent1), None)

        if parent2 is None:
            logger.info("  - Not enough distinct models in population to select a second parent.")
            return parent1, None

        return parent1, parent2
