"""
Main simulator for the M2N2 evolutionary experiment.

This module contains the `EvolutionSimulator` class, which orchestrates the
entire evolutionary process. It manages the simulation's configuration,
population, generational loop, and I/O operations like logging and model
saving.
"""
import torch
import os
import glob
import logging
import csv
import builtins

from .logger_config import setup_logger
from .model_wrapper import ModelWrapper
from .evolution import specialize, select_mates, merge, mutate, finetune, create_next_generation
from .data import get_dataloaders
from .visualization import plot_fitness_history
from .utils import set_seed
from .config_manager import ConfigManager
from .constants import COMMAND_FILE, FITNESS_LOG_FILENAME
from typing import List, Tuple, Type, Dict, Any
from .merge_strategies import (
    MergeStrategy, AverageMergeStrategy, FitnessWeightedMergeStrategy,
    LayerWiseMergeStrategy, SequentialConstructiveMergeStrategy
)
from .selection_strategies import MateSelectionStrategy, HealingMateSelectionStrategy
from .generation_strategies import GenerationStrategy, ReplaceWorstStrategy

logger = logging.getLogger("M2N2_SIMULATOR")

class EvolutionSimulator:
    """
    Encapsulates the entire evolutionary simulation.

    This class manages the lifecycle of the M2N2 experiment, including setup,
    running the generational loop, handling dynamic commands, and saving results.

    Attributes:
        config_manager (ConfigManager): Manages all simulation parameters.
        device (torch.device): The device (CPU or CUDA) for computation.
        population (List[ModelWrapper]): The current population of models.
        current_generation (int): The current generation number.
    """
    def __init__(self, config_path: str = 'config.yaml') -> None:
        """
        Initializes the simulator.

        Args:
            config_path (str, optional): Path to the configuration YAML file.
                Defaults to 'config.yaml'.
        """
        self.config_manager = ConfigManager(config_path)
        self._setup_environment()
        logger.info(f"--- M2N2 Simulation Initialized ---")
        logger.info(f"Model: {self.config_manager.model_name}, Device: {self.device}, Seed: {self.config_manager.seed}")
        self.model_dir = "src/pretrained_models"
        self._setup_simulation()

    def _setup_simulation(self) -> None:
        """Initializes or resets the simulation to its starting state."""
        self.population: List[ModelWrapper] = []
        self.fitness_history: List[Tuple[float, float]] = []
        self.current_generation = 0
        self.loaded_model_files: List[str] = []
        self._initialize_dataloaders()
        self._initialize_strategies()
        self._initialize_population()
        self._initialize_fitness_log()

    def _initialize_strategies(self) -> None:
        """Initializes strategy objects based on the configuration."""
        strategy_configs = [
            ('mate_selection', self.config_manager.mate_selection_strategy, {
                'healing': HealingMateSelectionStrategy
            }, {}),
            ('generation', self.config_manager.generation_strategy, {
                'replace_worst': ReplaceWorstStrategy
            }, {}),
            ('merge', self.config_manager.merge_strategy, {
                'average': AverageMergeStrategy,
                'fitness_weighted': FitnessWeightedMergeStrategy,
                'layer-wise': LayerWiseMergeStrategy,
                'sequential_constructive': SequentialConstructiveMergeStrategy
            }, {'seed': self.config_manager.seed if self.config_manager.merge_strategy == 'layer-wise' else None})
        ]

        for attr_name, strategy_name, mapping, kwargs in strategy_configs:
            strategy_instance = self._create_strategy(strategy_name, mapping, f"{attr_name} strategy", **kwargs)
            setattr(self, f"{attr_name}_strategy", strategy_instance)

    def _create_strategy(self, name: str, mapping: Dict[str, Type], type_str: str, **kwargs: Any) -> Any:
        """Factory helper to create a strategy instance."""
        strategy_class = mapping.get(name)
        if not strategy_class:
            raise ValueError(f"Unknown {type_str}: {name}")
        # Filter out None kwargs before passing to constructor
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return strategy_class(**filtered_kwargs)

    def _setup_environment(self) -> None:
        """Sets up the logger and computation device."""
        setup_logger(self.config_manager.config.get('log_file'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config_manager.seed)

    def _initialize_dataloaders(self) -> None:
        """Creates DataLoaders for the experiment."""
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
        """Initializes or loads the starting population of models."""
        logger.info("--- STEP 1: Initializing Population ---")
        model_files = glob.glob(os.path.join(self.model_dir, "*.pth"))
        if model_files:
            self._load_population_from_files(model_files)
        else:
            self._initialize_new_population()

    def _load_population_from_files(self, model_files: List[str]) -> None:
        """Loads a population from saved .pth files."""
        logger.info(f"Found {len(model_files)} models. Loading population.")
        for f in model_files:
            wrapper = ModelWrapper.from_file(
                f, self.config_manager.model_name, self.num_classes, self.device
            )
            if wrapper:
                self.population.append(wrapper)
                self.loaded_model_files.append(f)

    def _initialize_new_population(self) -> None:
        """Creates a new population of specialists from scratch."""
        logger.info("No pretrained models found. Initializing new population.")
        from .model_factory import create_model
        for i in range(self.config_manager.population_size):
            model = create_model(
                self.config_manager.model_name, self.num_classes, self.device
            )
            self.population.append(
                ModelWrapper(
                    model_name=self.config_manager.model_name,
                    model=model,
                    niche_classes=[i],
                    device=self.device
                )
            )
        logger.info("Specialization will occur in the first generation.")

    def _run_specialization_phase(self) -> None:
        """Trains specialist models on their respective niches."""
        logger.info("--- Specializing Models ---")
        for model in self.population:
            if model.niche_classes != list(range(self.num_classes)):
                specialize(model, self.config_manager)

    def _initialize_fitness_log(self) -> None:
        """Creates the fitness log file with a header."""
        if not os.path.exists(FITNESS_LOG_FILENAME):
            try:
                with builtins.open(FITNESS_LOG_FILENAME, 'w', newline='') as f:
                    csv.writer(f).writerow(['generation', 'best_fitness', 'average_fitness'])
            except OSError as e:
                logger.warning(f"Could not create fitness log: {e}")

    def _log_fitness_to_csv(self, gen: int, best: float, avg: float) -> None:
        """Appends fitness data for the current generation to the CSV log."""
        try:
            with builtins.open(FITNESS_LOG_FILENAME, 'a', newline='') as f:
                csv.writer(f).writerow([gen, best, avg])
        except OSError as e:
            logger.warning(f"Failed to write to fitness log: {e}")

    def _run_evaluation_phase(self) -> None:
        """Evaluates the fitness of the entire population."""
        if not self.population:
            logger.error("Population is empty. Cannot evaluate.")
            return
        logger.info("--- Evaluating Population ---")
        for model in self.population:
            model.evaluate(self.config_manager.dataset_name, self.config_manager.subset_percentage, self.config_manager.seed)
        best = max(m.fitness for m in self.population)
        avg = sum(m.fitness for m in self.population) / len(self.population)
        self.fitness_history.append((best, avg))
        logger.info(f"Generation {self.current_generation + 1} Stats: Best Fitness={best:.2f}%, Avg Fitness={avg:.2f}%")
        self._log_fitness_to_csv(self.current_generation + 1, best, avg)

    def _run_evolution_phase(self) -> None:
        """Creates the next generation through selection, crossover, and mutation."""
        logger.info("--- Mating and Evolution ---")
        if not self.population:
            logger.error("Population is empty. Skipping evolution.")
            return
        parent_pairs = select_mates(
            self.population, self.config_manager.num_offspring,
            self.mate_selection_strategy, self.config_manager
        )
        offspring = []
        for p1, p2 in parent_pairs:
            child = merge(p1, p2, self.merge_strategy, self.validation_loader)
            child = mutate(child, self.current_generation, self.config_manager, self.config_manager.seed)
            finetune(child, self.validation_loader, self.config_manager)
            offspring.append(child)
        if offspring:
            self.population = create_next_generation(
                self.population, offspring, self.generation_strategy, self.config_manager
            )

    def run_one_generation(self) -> None:
        """Runs a single generation of the simulation."""
        logger.info(f"\n--- GENERATION {self.current_generation + 1}/{self.config_manager.num_generations} ---")
        self._run_specialization_phase()
        self._run_evaluation_phase()
        self._run_evolution_phase()
        self.current_generation += 1

    def _handle_commands(self) -> str | None:
        """Checks for and handles dynamic commands from the dashboard."""
        cmds = self.config_manager.load_dynamic_config()
        if cmds:
            if any(k in cmds for k in ['mate_selection_strategy', 'generation_strategy', 'merge_strategy']):
                self._initialize_strategies()
            if cmds.get('restart_simulation'):
                self._restart()
                return "restart"
            if cmds.get('stop_simulation'):
                return "stop"
        return None

    def run(self) -> None:
        """Runs the main evolutionary loop."""
        while self.current_generation < self.config_manager.num_generations:
            cmd = self._handle_commands()
            if cmd == "restart": continue
            if cmd == "stop": break
            self.run_one_generation()
        self._summarize_and_save()

    def _summarize_and_save(self) -> None:
        """Prints a final summary and saves the final population."""
        if not self.population:
            logger.info("Simulation stopped with no population to summarize.")
            return
        logger.info("\n--- EXPERIMENT SUMMARY ---")
        best_model = max(self.population, key=lambda m: m.fitness)
        logger.info(f"Final best model accuracy: {best_model.fitness:.2f}%")
        plot_fitness_history(self.fitness_history, 'fitness_history.png')
        self._save_final_population()

    def _get_model_path(self, model_wrapper: ModelWrapper, model_dir: str) -> str:
        """Generates a standardized file path for a given model."""
        niche = "_".join(map(str, model_wrapper.niche_classes))
        return os.path.join(model_dir, f"model_niche_{niche}_fitness_{model_wrapper.fitness:.2f}.pth")

    def _save_final_population(self) -> None:
        """Saves the final population of models to disk."""
        logger.info(f"--- Saving final population to {self.model_dir} ---")
        os.makedirs(self.model_dir, exist_ok=True)
        if self.config_manager.delete_old_models:
            self._delete_old_models()
        for model in self.population:
            path = self._get_model_path(model, self.model_dir)
            try:
                model.save(path)
                logger.info(f"  - Saved model to {path}")
            except OSError as e:
                logger.warning(f"Failed to save model {path}: {e}")

    def _delete_old_models(self, preserve_loaded_models: bool = True) -> None:
        """
        Deletes generated model files from the model directory.
        Args:
            preserve_loaded_models (bool): If True, preserves files that were
                loaded at the start of the simulation. If False, deletes all
                models not in the final population.
        """
        logger.info(f"Clearing old models from {self.model_dir}...")
        final_files = {self._get_model_path(m, self.model_dir) for m in self.population}
        generated_files = set(glob.glob(os.path.join(self.model_dir, "*.pth")))
        files_to_delete = generated_files - final_files

        if preserve_loaded_models:
            files_to_delete -= set(self.loaded_model_files)

        for f in files_to_delete:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logger.info(f"  - Deleted old model: {f}")
            except OSError as e:
                logger.warning(f"Error deleting old model {f}: {e}")

    def _clear_simulation_artifacts(self) -> None:
        """Removes all generated files from the previous simulation run."""
        logger.info("Clearing previous simulation artifacts...")
        for path in [FITNESS_LOG_FILENAME, COMMAND_FILE, 'fitness_history.png']:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"  - Removed {path}")
            except OSError as e:
                logger.warning(f"Error removing artifact {path}: {e}")
        self._delete_old_models(preserve_loaded_models=False)

    def _restart(self) -> None:
        """Resets the simulation to its initial state."""
        logger.info("\n--- RESTARTING SIMULATION ---")
        self._clear_simulation_artifacts()
        self._setup_simulation()
        logger.info("--- Simulation has been restarted ---")
