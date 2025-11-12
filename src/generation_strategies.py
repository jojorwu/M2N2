"""
Defines the strategies for creating the next generation in the simulation.

This module implements the "survival of the fittest" phase of the evolutionary
algorithm. Generation strategies determine which models from the current
population and the newly created offspring pool will survive to form the next
generation. This allows for different selection mechanisms, such as elitism.
"""
from abc import ABC, abstractmethod
from typing import List

# Forward-declare for type hinting
if "ConfigManager" not in globals():
    from typing import TypeVar
    ConfigManager = TypeVar("ConfigManager")

import logging
from .model_wrapper import ModelWrapper

logger = logging.getLogger("M2N2_SIMULATOR")

class GenerationStrategy(ABC):
    """
    Abstract base class for defining a generation survival strategy.

    This class provides the interface for all generation strategies. Subclasses
    must implement the `create_next_generation` method, which contains the
    core logic for selecting the members of the next generation.
    """
    @abstractmethod
    def create_next_generation(
        self,
        current_population: List[ModelWrapper],
        offspring_pool: List[ModelWrapper],
        config_manager: "ConfigManager"
    ) -> List[ModelWrapper]:
        """
        Creates the next generation's population from the current one and offspring.

        Args:
            current_population (List[ModelWrapper]): The list of models from
                the current generation.
            offspring_pool (List[ModelWrapper]): The list of newly created and
                trained offspring.
            config_manager (ConfigManager): The simulation's configuration object,
                providing access to parameters like `population_size`.

        Returns:
            List[ModelWrapper]: A list of `ModelWrapper` objects representing
                the selected population for the next generation.
        """
        pass

class ReplaceWorstStrategy(GenerationStrategy):
    """
    An elitist selection strategy that ensures the population's best models survive.

    This strategy combines the current population with the new offspring, evaluates
    the fitness of all individuals in this combined pool, and selects the top
    `population_size` models to form the next generation. This guarantees that
    high-performing models are always carried over, preventing regression in the
    population's overall fitness.
    """
    def create_next_generation(
        self,
        current_population: List[ModelWrapper],
        offspring_pool: List[ModelWrapper],
        config_manager: "ConfigManager"
    ) -> List[ModelWrapper]:
        """
        Creates the next generation by selecting the fittest individuals from a
        combined pool of the current population and new offspring.

        The process involves:
        1. Evaluating the fitness of all new offspring.
        2. Combining the current population and the evaluated offspring.
        3. Sorting the combined pool by fitness in descending order.
        4. Selecting the top `population_size` models for the next generation.

        Args:
            current_population (List[ModelWrapper]): The list of models from
                the current generation.
            offspring_pool (List[ModelWrapper]): The list of newly created
                offspring.
            config_manager (ConfigManager): The simulation's configuration, used
                to get the target `population_size`.

        Returns:
            List[ModelWrapper]: The list of the fittest models that will form
                the next generation.
        """
        logger.info("Creating next generation using 'Replace Worst' (Elitist) strategy...")

        # Evaluate all new offspring to ensure their fitness is up-to-date
        for child in offspring_pool:
            if not child.fitness_is_current:
                child.evaluate(
                    dataset_name=config_manager.dataset_name,
                    subset_percentage=config_manager.subset_percentage,
                    seed=config_manager.seed
                )
                logger.info(f"  - New offspring evaluated with fitness: {child.fitness:.2f}%")

        # Combine the old population with the new offspring
        full_pool = list(current_population) + offspring_pool

        # Sort the entire pool by fitness in descending order
        full_pool.sort(key=lambda x: x.fitness, reverse=True)

        # The next generation consists of the top 'population_size' individuals
        next_generation = full_pool[:config_manager.population_size]

        logger.info(f"Selected {len(next_generation)} fittest individuals for the next generation.")
        return next_generation
