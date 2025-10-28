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
        dataset_name: str,
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
        dataset_name: str,
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
        min_accuracy = min(class_accuracies)
        weakest_indices = [i for i, acc in enumerate(class_accuracies) if acc == min_accuracy]
        logger.info(f"  - Parent 1's weakest classes are {weakest_indices} (Accuracy: {min_accuracy:.2f}%)")

        parent2 = None
        # Shuffle the weakest indices to randomize the search order.
        random.shuffle(weakest_indices)

        for class_index in weakest_indices:
            specialist_candidates = [
                m for m in population if m.niche_classes == [class_index] and m is not parent1
            ]
            if specialist_candidates:
                parent2 = max(specialist_candidates, key=lambda m: m.fitness)
                logger.info(f"  - Found specialist for class {class_index} as Parent 2 (Fitness: {parent2.fitness:.2f}%)")
                break # Found the best available specialist, so we can stop.
        else:
            logger.info("  - No suitable specialist found. Using second-best model as fallback Parent 2.")
            sorted_population = sorted(population, key=lambda m: m.fitness, reverse=True)
            parent2 = next((model for model in sorted_population if model is not parent1 and model != parent1), None)

        if parent2 is None:
            logger.info("  - Not enough distinct models in population to select a second parent.")
            return parent1, None

        return parent1, parent2
