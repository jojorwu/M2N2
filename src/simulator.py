"""
This module contains the EvolutionSimulator class, which encapsulates the
entire logic for running an M2N2-inspired evolutionary experiment.
"""
import torch
import os
import glob
import logging
import csv
import builtins

from .logger_config import setup_logger
from .model_wrapper import ModelWrapper
from .model_factory import create_model
from .evolution import specialize, select_mates, merge, mutate, finetune, create_next_generation
from .data import get_dataloaders
from .visualization import plot_fitness_history
from .utils import set_seed
from .config_manager import ConfigManager
from .constants import COMMAND_FILE, FITNESS_LOG_FILENAME
from typing import List, Tuple, Type, Dict, Any
from torch.utils.data import DataLoader
from .merge_strategies import (
    AverageMergeStrategy,
    FitnessWeightedMergeStrategy,
    LayerWiseMergeStrategy,
    SequentialConstructiveMergeStrategy,
)
from .selection_strategies import HealingMateSelectionStrategy
from .generation_strategies import ReplaceWorstStrategy

logger = logging.getLogger("M2N2_SIMULATOR")

class EvolutionSimulator:
    """
    Encapsulates the entire evolutionary simulation, from configuration
    loading to running the generational loop and saving the results.
    """
    def __init__(self, config_path: str = 'config.yaml') -> None:
        """
        Initializes the simulator by loading configuration and setting up
        the environment.

        Args:
            config_path (str, optional): The path to the configuration YAML
                file. Defaults to 'config.yaml'.
        """
        self.config_manager = ConfigManager(config_path)
        self._setup_environment()

        logger.info("--- M2N2 Simplified Implementation ---")
        logger.info(f"Loaded configuration for model: {self.config_manager.model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Experiment seed: {self.config_manager.seed}\n")

        self.population: List[ModelWrapper] = []
        self.fitness_history: List[Tuple[float, float]] = []
        self.current_generation = 0
        self.loaded_model_files: List[str] = []

        self._initialize_dataloaders()
        self._initialize_strategies()
        self._initialize_population()
        self._initialize_fitness_log()

    def _initialize_strategies(self) -> None:
        """Initializes the strategy objects based on the configuration using a data-driven approach."""
        strategy_configs = [
            {
                'name': 'mate_selection',
                'attribute': 'mate_selection_strategy',
                'config_key': 'mate_selection_strategy',
                'map': {'healing': HealingMateSelectionStrategy},
                'args': {}
            },
            {
                'name': 'generation',
                'attribute': 'generation_strategy',
                'config_key': 'generation_strategy',
                'map': {'replace_worst': ReplaceWorstStrategy},
                'args': {}
            },
            {
                'name': 'merge',
                'attribute': 'merge_strategy',
                'config_key': 'merge_strategy',
                'map': {
                    'average': AverageMergeStrategy,
                    'fitness_weighted': FitnessWeightedMergeStrategy,
                    'layer-wise': LayerWiseMergeStrategy,
                    'sequential_constructive': SequentialConstructiveMergeStrategy,
                },
                'args': {
                    'layer-wise': {'seed': self.config_manager.seed}
                }
            }
        ]

        for config in strategy_configs:
            strategy_name = getattr(self.config_manager, config['config_key'])
            strategy_args = config['args'].get(strategy_name, {})
            strategy_instance = self._create_strategy(
                strategy_name,
                config['map'],
                config['name'],
                **strategy_args
            )
            setattr(self, config['attribute'], strategy_instance)

    def _create_strategy(self, strategy_name: str, strategy_map: Dict[str, Type], strategy_type: str, **kwargs: Any) -> Any:
        """
        Factory helper to create a strategy instance.

        Args:
            strategy_name (str): The name of the strategy from the config.
            strategy_map (Dict[str, Type]): A map from name to class.
            strategy_type (str): A string descriptor for the strategy type,
                used for error messages (e.g., "mate selection").
            **kwargs: Additional keyword arguments to pass to the strategy's
                constructor.

        Returns:
            An instance of the requested strategy.

        Raises:
            ValueError: If the strategy_name is not found in the map.
        """
        strategy_class = strategy_map.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown {strategy_type} strategy: {strategy_name}")
        return strategy_class(**kwargs)

    def _setup_environment(self) -> None:
        """Sets up the logger and device."""
        log_file = self.config_manager.config.get('log_file')
        setup_logger(log_file=log_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_dataloaders(self) -> None:
        """Creates the necessary DataLoaders for the experiment."""
        logger.info("--- Creating DataLoaders ---")
        _, self.validation_loader, _, self.num_classes = get_dataloaders(
            dataset_name=self.config_manager.dataset_name,
            model_name=self.config_manager.model_name,
            batch_size=self.config_manager.batch_size,
            subset_percentage=self.config_manager.subset_percentage,
            validation_split=self.config_manager.validation_split,
            seed=self.config_manager.seed
        )

    def _initialize_population(self) -> None:
        """Initializes or loads the population of models."""
        if self.config_manager.seed is not None:
            set_seed(self.config_manager.seed)
        logger.info("--- STEP 1: Initializing or Loading Population ---")
        model_dir = "src/pretrained_models"
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if model_files:
            self._load_population_from_files(model_files)
        else:
            self._initialize_new_population()

    def _load_population_from_files(self, model_files: List[str]) -> None:
        """
        Loads a population of models from saved .pth files using the
        ModelWrapper's factory method.
        """
        logger.info(f"Found {len(model_files)} models in 'src/pretrained_models'. Attempting to load them.")
        for f in model_files:
            wrapper = ModelWrapper.from_file(
                filepath=f,
                model_name=self.config_manager.model_name,
                num_classes=self.num_classes,
                device=self.device
            )
            if wrapper:
                self.population.append(wrapper)
                self.loaded_model_files.append(f)

    def _initialize_new_population(self) -> None:
        """Initializes a new population from scratch and specializes them."""
        logger.info("No pretrained models found. Initializing a new population from scratch.")
        niches = [[i] for i in range(self.config_manager.population_size)]
        for i in range(self.config_manager.population_size):
            model = create_model(
                model_name=self.config_manager.model_name,
                num_classes=self.num_classes,
                device=self.device
            )
            self.population.append(
                ModelWrapper(
                    model=model,
                    device=self.device,
                    model_name=self.config_manager.model_name,
                    niche_classes=niches[i]
                )
            )

        logger.info("--- Initial population created. Specialization will occur in the first generation. ---")

    def _run_specialization_phase(self, generation: int) -> None:
        """Handles the specialization of models in the population."""
        logger.info("--- Specializing Models ---")
        for model_wrapper in self.population:
            if model_wrapper.niche_classes != list(range(self.num_classes)):
                specialize(model_wrapper, self.config_manager)
        logger.info("")

    def _initialize_fitness_log(self):
        """Creates the fitness log file and writes the header if it doesn't exist."""
        try:
            if not os.path.exists(FITNESS_LOG_FILENAME):
                with builtins.open(FITNESS_LOG_FILENAME, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['generation', 'best_fitness', 'average_fitness'])
        except OSError as e:
            logger.warning(f"Could not write to fitness log file at {FITNESS_LOG_FILENAME}: {e}")

    def _log_fitness_to_csv(self, generation: int, best_fitness: float, average_fitness: float):
        """Appends the fitness data for the current generation to the CSV log."""
        try:
            with builtins.open(FITNESS_LOG_FILENAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([generation, best_fitness, average_fitness])
        except OSError as e:
            logger.warning(f"Failed to append to fitness log file at {FITNESS_LOG_FILENAME}: {e}")

    def _run_evaluation_phase(self) -> None:
        """Handles the evaluation of the population."""
        if not self.population:
            logger.error("Population is empty. Cannot run evaluation.")
            return

        logger.info("--- Evaluating Population on Test Set ---")
        for model_wrapper in self.population:
            model_wrapper.evaluate(
                dataset_name=self.config_manager.dataset_name,
                subset_percentage=self.config_manager.subset_percentage,
                seed=self.config_manager.seed
            )

        best_fitness = max([m.fitness for m in self.population])
        avg_fitness = sum([m.fitness for m in self.population]) / len(self.population)
        self.fitness_history.append((best_fitness, avg_fitness))
        generation = len(self.fitness_history)
        logger.info(f"\nGeneration {generation} Stats: Best Fitness = {best_fitness:.2f}%, Avg Fitness = {avg_fitness:.2f}%\n")
        self._log_fitness_to_csv(generation, best_fitness, avg_fitness)

    def _clear_simulation_artifacts(self) -> None:
        """Clears logs and saved models from previous runs."""
        logger.info("--- Clearing simulation artifacts ---")
        if os.path.exists(FITNESS_LOG_FILENAME):
            try:
                os.remove(FITNESS_LOG_FILENAME)
                logger.info(f"Removed {FITNESS_LOG_FILENAME}")
            except OSError as e:
                logger.warning(f"Error removing log file {FITNESS_LOG_FILENAME}: {e}")

        model_dir = "src/pretrained_models"
        if os.path.exists(model_dir):
            pattern = os.path.join(model_dir, "model_niche_*.pth")
            files = glob.glob(pattern)
            if files:
                cleared_count = 0
                for f in files:
                    try:
                        os.remove(f)
                        cleared_count += 1
                    except OSError as e:
                        logger.warning(f"Error removing model file {f}: {e}")
                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} models from {model_dir}")

        if os.path.exists(COMMAND_FILE):
            try:
                os.remove(COMMAND_FILE)
                logger.info(f"Removed {COMMAND_FILE}")
            except OSError as e:
                logger.warning(f"Error removing command file {COMMAND_FILE}: {e}")

    def _restart(self) -> None:
        """Resets the simulation to its initial state for a fresh run."""
        logger.info("\n--- RESTARTING SIMULATION ---")
        self._clear_simulation_artifacts()
        self.population = []
        self.fitness_history = []
        self.current_generation = 0
        self._initialize_population()
        self._initialize_fitness_log()
        logger.info("--- Simulation has been restarted ---")

    def _run_evolution_phase(self, generation: int) -> None:
        """Handles the mating, mutation, and selection of models."""
        logger.info("--- Mating and Evolution ---")

        if not self.population:
            logger.error("Population is empty. Skipping evolution phase.")
            return

        parent_pairs = select_mates(
            self.population,
            num_pairs=self.config_manager.num_offspring,
            strategy=self.mate_selection_strategy,
            config_manager=self.config_manager
        )

        offspring_pool = []
        for i, (parent1, parent2) in enumerate(parent_pairs):
            logger.info(f"--- Creating Offspring {i+1}/{len(parent_pairs)} ---")
            child = merge(
                parent1, parent2,
                strategy=self.merge_strategy,
                validation_loader=self.validation_loader
            )
            child = mutate(
                child,
                generation=generation,
                config_manager=self.config_manager,
                seed=self.config_manager.seed
            )
            finetune(
                child,
                validation_loader=self.validation_loader,
                config_manager=self.config_manager
            )
            offspring_pool.append(child)

        if offspring_pool:
            self.population = create_next_generation(
                self.population,
                offspring_pool,
                strategy=self.generation_strategy,
                config_manager=self.config_manager
            )
        else:
            logger.info("No offspring were created. Population will carry over to the next generation without changes.")

    def run_one_generation(self) -> None:
        """Runs a single generation of the evolutionary simulation."""
        logger.info(f"\n--- GENERATION {self.current_generation + 1}/{self.config_manager.num_generations} ---")
        self._run_specialization_phase(self.current_generation)
        self._run_evaluation_phase()
        self._run_evolution_phase(self.current_generation)
        self.current_generation += 1

    def _handle_commands(self) -> str | None:
        """
        Checks for and handles dynamic commands from command_config.json,
        returning a string command if action is needed.
        """
        command_config = self.config_manager.load_dynamic_config()

        if command_config:  # If any dynamic config was loaded
            # Check if strategy-related keys were changed and re-initialize if so
            strategy_keys = ['mate_selection_strategy', 'generation_strategy', 'merge_strategy']
            if any(key in command_config for key in strategy_keys):
                logger.info("Re-initializing strategies due to dynamic configuration change.")
                self._initialize_strategies()

        if command_config.get('restart_simulation'):
            self._restart()
            return "restart"

        if command_config.get('stop_simulation'):
            logger.info("Stop command received. Shutting down gracefully.")
            return "stop"

        return None

    def run(self) -> None:
        """Runs the main evolutionary loop, checking for commands each generation."""
        while self.current_generation < self.config_manager.num_generations:
            command = self._handle_commands()
            if command == "restart":
                continue
            if command == "stop":
                break

            self.run_one_generation()

        if self.fitness_history:
            self._summarize_and_save()
        else:
            logger.info("Simulation stopped before any generations were completed. No summary to generate.")

    def _summarize_and_save(self) -> None:
        """Prints a final summary and saves the final population."""
        logger.info("\n\n--- EXPERIMENT SUMMARY ---")
        logger.info("Fitness history (Best, Average):")
        for i, (best, avg) in enumerate(self.fitness_history):
            logger.info(f"  - Generation {i+1}: Best={best:.2f}%, Avg={avg:.2f}%")

        final_best_model = max(self.population, key=lambda m: m.fitness)
        logger.info(f"\nFinal best model achieved an accuracy of {final_best_model.fitness:.2f}%")

        plot_fitness_history(self.fitness_history, 'fitness_history.png')
        self._save_final_population()

    def _save_final_population(self, model_dir: str = "src/pretrained_models") -> None:
        """
        Saves the final population of models to disk.

        Args:
            model_dir (str, optional): The directory to save models to.
                Defaults to "src/pretrained_models".
        """
        logger.info(f"\n--- Saving final population to {model_dir}/ ---")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if self.config_manager.delete_old_models:
            self._delete_old_models(model_dir)
        else:
            logger.info("`delete_old_models` is false. Skipping cleanup of old models.")

        for model_wrapper in self.population:
            niche_str = "_".join(map(str, model_wrapper.niche_classes))
            filename = f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth"
            model_path = os.path.join(model_dir, filename)
            model_wrapper.save(model_path)
            logger.info(f"  - Saved model to {model_path}")

    def _delete_old_models(self, model_dir: str):
        """
        Deletes all model files from the output directory that are not part of
        the final, surviving population. This includes models from previous
        generations and any initial models that were replaced.
        """
        if not os.path.isdir(model_dir):
            return

        logger.info(f"Clearing old models from {model_dir}...")

        # 1. Determine the full paths of the models to keep
        files_to_keep = set()
        for model_wrapper in self.population:
            niche_str = "_".join(map(str, model_wrapper.niche_classes))
            filename = f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth"
            files_to_keep.add(os.path.join(model_dir, filename))

        # 2. Get all potential model files to check:
        #    - Files from intermediate generations matching the pattern.
        #    - Files that were loaded at the start of the simulation.
        intermediate_files = glob.glob(os.path.join(model_dir, "model_niche_*.pth"))
        all_files_to_check = set(intermediate_files) | set(self.loaded_model_files)

        # 3. Delete any file that is not in the 'keep' list
        deleted_count = 0
        for filepath in all_files_to_check:
            if filepath not in files_to_keep:
                try:
                    # Check existence as a loaded file might have been deleted
                    # if it also matched the glob pattern and was handled already.
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.info(f"Deleted old model file: {filepath}")
                        deleted_count += 1
                except OSError as e:
                    logger.warning(f"Error deleting file {filepath}: {e}")

        if deleted_count == 0:
            logger.info("No old models found to clear.")
