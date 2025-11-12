"""
Implements mate selection strategies for the evolutionary algorithm.

This module uses the Strategy design pattern to decouple the main evolutionary
algorithm from the specific implementation of how parents are selected for
mating. Each strategy is a class that implements a common interface, allowing
for different selection heuristics (e.g., random, fitness-based, "healing")
to be easily interchanged.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

# Forward-declare for type hinting
if "ConfigManager" not in globals():
    from typing import TypeVar
    ConfigManager = TypeVar("ConfigManager")

import logging
from .model_wrapper import ModelWrapper

logger = logging.getLogger("M2N2_SIMULATOR")

class MateSelectionStrategy(ABC):
    """
    Abstract base class for all mate selection strategies.

    This class defines the interface that all selection strategy implementations
    must follow. Subclasses are required to implement the `select_parent_pairs`
    method.
    """
    @abstractmethod
    def select_parent_pairs(
        self,
        population: List[ModelWrapper],
        num_pairs: int,
        config_manager: "ConfigManager"
    ) -> List[Tuple[ModelWrapper, ModelWrapper]]:
        """
        Selects multiple pairs of parents from the population for breeding.

        Args:
            population (List[ModelWrapper]): The current population of models
                from which to select parents.
            num_pairs (int): The target number of parent pairs to create.
            config_manager (ConfigManager): The simulation's configuration object.

        Returns:
            List[Tuple[ModelWrapper, ModelWrapper]]: A list of tuples, where each
                tuple contains two parent `ModelWrapper` objects.
        """
        pass

class HealingMateSelectionStrategy(MateSelectionStrategy):
    """
    Selects parents using a "healing" strategy to improve weaknesses.

    This strategy aims to produce offspring that are better generalists by
    intelligently pairing models. It identifies the best-performing model in the
    population as the primary parent. It then analyzes this parent to find its
    weakest performing classes and pairs it with specialist models that are
    experts in those specific classes. If not enough specialists are available,
    it falls back to pairing the primary parent with other distinct, high-
    performing models.
    """
    def select_parent_pairs(
        self,
        population: List[ModelWrapper],
        num_pairs: int,
        config_manager: "ConfigManager"
    ) -> List[Tuple[ModelWrapper, ModelWrapper]]:
        """
        Selects parent pairs by pairing the best model with complementary specialists.

        The process is as follows:
        1. Identify the single best model in the population as `parent1`.
        2. Evaluate `parent1`'s accuracy on each individual class to find its weaknesses.
        3. Iterate through the weaknesses, from worst to best.
        4. For each weak class, find the best available specialist model for that
           class to serve as a partner.
        5. If the target number of pairs is not met, fall back to pairing `parent1`
           with other high-performing (but genetically distinct) models.

        Args:
            population (List[ModelWrapper]): The current population of models.
            num_pairs (int): The number of parent pairs to select.
            config_manager (ConfigManager): The simulation's configuration.

        Returns:
            A list of tuples, each containing `parent1` and a selected partner.
        """
        logger.info(f"Selecting {num_pairs} parent pairs with healing strategy...")
        if len(population) < 2:
            return []

        parent1 = max(population, key=lambda m: m.fitness)
        logger.info(f"  - Primary Parent: Best model (Fitness: {parent1.fitness:.2f}%)")

        class_accuracies = parent1.evaluate_by_class(
            dataset_name=config_manager.dataset_name,
            subset_percentage=config_manager.subset_percentage,
            seed=config_manager.seed
        )
        sorted_weak_classes = sorted(range(len(class_accuracies)), key=class_accuracies.__getitem__)

        parent_pairs = []
        used_partners = {parent1}

        # Phase 1: Find Specialist Partners for weakest classes
        for class_index in sorted_weak_classes:
            if len(parent_pairs) >= num_pairs: break
            specialist = next((m for m in sorted(population, key=lambda x: x.fitness, reverse=True) if m.niche_classes == [class_index] and m not in used_partners), None)
            if specialist:
                parent_pairs.append((parent1, specialist))
                used_partners.add(specialist)
                logger.info(f"    - Found specialist for weak class {class_index}. Pair created.")

        # Phase 2: Fallback to other high-performing partners if needed
        if len(parent_pairs) < num_pairs:
            logger.info("  - Not enough specialists found, falling back to other high-performers.")
            general_candidates = sorted([m for m in population if m not in used_partners], key=lambda m: m.fitness, reverse=True)
            for partner in general_candidates:
                if len(parent_pairs) >= num_pairs: break
                parent_pairs.append((parent1, partner))
                used_partners.add(partner)
                logger.info("    - Selected a high-performing generalist as a partner.")

        if len(parent_pairs) < num_pairs:
            logger.warning(f"  - Could only form {len(parent_pairs)} unique pairs out of {num_pairs} requested.")

        return parent_pairs
