"""
Implements various mate selection strategies for the evolutionary algorithm.

This module uses the Strategy design pattern to decouple the main
evolutionary algorithm from the specific implementation of how parents are
selected for mating. Each strategy is a class that implements a common
interface.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import logging

from .model_wrapper import ModelWrapper

logger = logging.getLogger("M2N2_SIMULATOR")

class MateSelectionStrategy(ABC):
    """Abstract base class for all mate selection strategies."""
    @abstractmethod
    def select_parent_pairs(
        self,
        population: List[ModelWrapper],
        num_pairs: int,
        config_manager: "ConfigManager"
    ) -> List[Tuple[ModelWrapper, ModelWrapper]]:
        """Selects multiple pairs of parents from the population."""
        pass

class HealingMateSelectionStrategy(MateSelectionStrategy):
    """
    Selects parents using a "healing" strategy.

    This strategy pairs the strongest overall model with specialists that are
    experts in the strongest model's weakest areas. If not enough specialists
    can be found, it falls back to pairing the strongest model with other
    high-performing (but genetically distinct) models.
    """
    def select_parent_pairs(
        self,
        population: List[ModelWrapper],
        num_pairs: int,
        config_manager: "ConfigManager"
    ) -> List[Tuple[ModelWrapper, ModelWrapper]]:
        logger.info(f"Selecting {num_pairs} parent pairs with healing strategy...")
        if len(population) < 2:
            logger.warning("  - Not enough models in population to form any pairs.")
            return []

        parent1 = max(population, key=lambda m: m.fitness)
        logger.info(f"  - Primary Parent (Parent 1) is the population's best model (Fitness: {parent1.fitness:.2f}%)")

        logger.info("  - Analyzing Parent 1's performance by class...")
        class_accuracies = parent1.evaluate_by_class(
            dataset_name=config_manager.dataset_name,
            subset_percentage=config_manager.subset_percentage,
            seed=config_manager.seed
        )

        # Get all weakest classes, sorted, to search through
        sorted_class_indices = sorted(range(len(class_accuracies)), key=lambda k: class_accuracies[k])
        logger.info(f"  - Parent 1's weakest classes (sorted): {sorted_class_indices}")

        parent_pairs: List[Tuple[ModelWrapper, ModelWrapper]] = []
        used_partners = {parent1}

        # --- Phase 1: Find Specialist Partners ---
        logger.info("  - Phase 1: Searching for specialist partners...")
        for class_index in sorted_class_indices:
            if len(parent_pairs) >= num_pairs:
                break

            specialist_candidates = [
                m for m in population if m.niche_classes == [class_index] and m not in used_partners
            ]

            if specialist_candidates:
                partner = max(specialist_candidates, key=lambda m: m.fitness)
                parent_pairs.append((parent1, partner))
                used_partners.add(partner)
                logger.info(f"    - Found specialist for weak class {class_index}. Pair {len(parent_pairs)}/{num_pairs} created.")

        logger.info(f"  - Found {len(parent_pairs)} specialist partners.")

        # --- Phase 2: Fallback to General High-Performing Partners ---
        if len(parent_pairs) < num_pairs:
            logger.info("  - Phase 2: Not enough specialists. Falling back to other high-performing models.")

            # Get a list of generalist candidates, sorted by fitness
            general_candidates = sorted(
                [m for m in population if m not in used_partners],
                key=lambda m: m.fitness,
                reverse=True
            )

            for partner in general_candidates:
                if len(parent_pairs) >= num_pairs:
                    break
                parent_pairs.append((parent1, partner))
                used_partners.add(partner)
                logger.info(f"    - Selected high-performer. Pair {len(parent_pairs)}/{num_pairs} created.")

        if len(parent_pairs) < num_pairs:
            logger.warning(f"  - Could only form {len(parent_pairs)} unique pairs, which is less than the requested {num_pairs}.")

        return parent_pairs
