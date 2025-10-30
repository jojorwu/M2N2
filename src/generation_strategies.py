"""
This module defines the strategies for creating the next generation of models
in the evolutionary simulation.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import logging
from .model_wrapper import ModelWrapper

logger = logging.getLogger("M2N2_SIMULATOR")

class GenerationStrategy(ABC):
    """
    Abstract base class for defining a generation strategy.
    """
    @abstractmethod
    def create_next_generation(
        self,
        current_population: List[ModelWrapper],
        new_child: ModelWrapper,
        config_manager: "ConfigManager"
    ) -> List[ModelWrapper]:
        """
        Creates the next generation's population.
        """
        pass

class ReplaceWorstStrategy(GenerationStrategy):
    """
    An elitist selection strategy where the new child replaces the worst-
    performing model in the population, if it is better.
    """
    def create_next_generation(
        self,
        current_population: List[ModelWrapper],
        new_child: ModelWrapper,
        config_manager: "ConfigManager"
    ) -> List[ModelWrapper]:
        """
        Creates the next generation by replacing the worst model if the
        new child has a higher fitness.
        """
        logger.info("Creating the next generation using 'Replace Worst' strategy...")

        # Evaluate the new child to make sure its fitness is calculated
        new_child.evaluate(
            dataset_name=config_manager.dataset_name,
            subset_percentage=config_manager.subset_percentage,
            seed=config_manager.seed
        )

        # Combine the old population with the new child, avoiding duplicates
        if new_child in current_population:
            logger.info("  - New child is a duplicate of an existing model. Not adding to the pool.")
            full_pool = current_population
        else:
            full_pool = current_population + [new_child]

        # Sort the entire pool by fitness in descending order
        full_pool.sort(key=lambda x: x.fitness, reverse=True)

        # The next generation consists of the top 'population_size' individuals
        next_generation = full_pool[:config_manager.population_size]

        logger.info(f"Selected {len(next_generation)} fittest individuals for the next generation.")

        return next_generation
