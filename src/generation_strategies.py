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
        offspring_pool: List[ModelWrapper],
        config_manager: "ConfigManager"
    ) -> List[ModelWrapper]:
        """
        Creates the next generation's population.
        """
        pass

class ReplaceWorstStrategy(GenerationStrategy):
    """
    An elitist selection strategy where the new offspring replace the worst-
    performing models in the population, if they are better.
    """
    def create_next_generation(
        self,
        current_population: List[ModelWrapper],
        offspring_pool: List[ModelWrapper],
        config_manager: "ConfigManager"
    ) -> List[ModelWrapper]:
        """
        Creates the next generation by combining the current population with
        the new offspring and selecting the fittest individuals.
        """
        logger.info("Creating the next generation using 'Replace Worst' strategy...")

        # Evaluate all new offspring
        for child in offspring_pool:
            child.evaluate(
                dataset_name=config_manager.dataset_name,
                subset_percentage=config_manager.subset_percentage,
                seed=config_manager.seed
            )
            logger.info(f"  - New offspring evaluated with fitness: {child.fitness:.2f}%")

        # Combine the old population with the new offspring, avoiding duplicates
        full_pool = list(current_population)
        for child in offspring_pool:
            if child in full_pool:
                logger.info(f"  - Offspring (fitness: {child.fitness:.2f}%) is a duplicate. Not adding to the pool.")
            else:
                full_pool.append(child)

        # Sort the entire pool by fitness in descending order
        full_pool.sort(key=lambda x: x.fitness, reverse=True)

        # The next generation consists of the top 'population_size' individuals
        next_generation = full_pool[:config_manager.population_size]

        logger.info(f"Selected {len(next_generation)} fittest individuals for the next generation.")

        return next_generation
