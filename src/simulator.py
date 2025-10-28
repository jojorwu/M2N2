"""
This module contains the EvolutionSimulator class, which encapsulates the
entire logic for running an M2N2-inspired evolutionary experiment.
"""
import torch
import os
import glob
import re
import logging
from .logger_config import setup_logger
from .model_wrapper import ModelWrapper
from .evolution import specialize, select_mates, merge, mutate, finetune, create_next_generation
from .data import get_dataloaders
from .visualization import plot_fitness_history
from .utils import set_seed
from .config_manager import ConfigManager
from typing import List, Tuple
from torch.utils.data import DataLoader

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
        logger.info(f"Loaded configuration for model: {self.config_manager.model_config}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Experiment seed: {self.config_manager.seed}\n")

        self.population: List[ModelWrapper] = []
        self.fitness_history: List[Tuple[float, float]] = []
        self.current_generation = 0
        self.loaded_model_files: List[str] = []

        self._initialize_dataloaders()
        self._initialize_population()
        self._initialize_fitness_log()

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
            model_name=self.config_manager.model_config,
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
        Loads a population of models from saved .pth files.
        """
        logger.info(f"Found {len(model_files)} models in 'src/pretrained_models'. Attempting to load them.")
        for f in model_files:
            match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(f))
            if match:
                niche_classes = [int(n) for n in match.group(1).split('_')]
                fitness = float(match.group(2))
                wrapper = ModelWrapper(
                    model_name=self.config_manager.model_config,
                    niche_classes=niche_classes,
                    device=self.device,
                    num_classes=self.num_classes
                )
                wrapper.model.load_state_dict(torch.load(f, map_location=self.device))
                wrapper.fitness = fitness
                wrapper.fitness_is_current = False
                self.population.append(wrapper)
                self.loaded_model_files.append(f)

    def _initialize_new_population(self) -> None:
        """Initializes a new population from scratch and specializes them."""
        logger.info("No pretrained models found. Initializing a new population from scratch.")
        niches = [[i] for i in range(self.config_manager.population_size)]
        for i in range(self.config_manager.population_size):
            self.population.append(
                ModelWrapper(
                    model_name=self.config_manager.model_config,
                    niche_classes=niches[i],
                    device=self.device,
                    num_classes=self.num_classes
                )
            )

        logger.info("--- Specializing Initial Models ---")
        for model_wrapper in self.population:
            specialize(
                model_wrapper,
                dataset_name=self.config_manager.dataset_name,
                epochs=self.config_manager.specialize_epochs,
                precision=self.config_manager.precision_config,
                seed=self.config_manager.seed,
                learning_rate=self.config_manager.learning_rate,
                subset_percentage=self.config_manager.subset_percentage,
                show_progress_bar=self.config_manager.show_progress_bar
            )
        logger.info("")

    def _run_specialization_phase(self, generation: int) -> None:
        """Handles the specialization of models in the population."""
        if generation > 0:
            logger.info("--- Specializing Models ---")
            for model_wrapper in self.population:
                if model_wrapper.niche_classes != list(range(self.num_classes)):
                    specialize(
                        model_wrapper,
                        dataset_name=self.config_manager.dataset_name,
                        epochs=self.config_manager.specialize_epochs,
                        precision=self.config_manager.precision_config,
                        seed=self.config_manager.seed,
                        learning_rate=self.config_manager.learning_rate,
                        subset_percentage=self.config_manager.subset_percentage
                    )
            logger.info("")

    def _initialize_fitness_log(self) -> None:
        """Creates the fitness log file and writes the header."""
        with open("fitness_log.csv", "w") as f:
            f.write("generation,best_fitness,average_fitness\n")

    def _log_fitness_to_csv(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        """Appends the current generation's fitness data to the CSV log."""
        with open("fitness_log.csv", "a") as f:
            f.write(f"{generation},{best_fitness:.2f},{avg_fitness:.2f}\n")

    def _run_evaluation_phase(self) -> None:
        """Handles the evaluation of the population."""
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
        if os.path.exists("fitness_log.csv"):
            os.remove("fitness_log.csv")
            logger.info("Removed fitness_log.csv")

        model_dir = "src/pretrained_models"
        if os.path.exists(model_dir):
            pattern = os.path.join(model_dir, "model_niche_*.pth")
            files = glob.glob(pattern)
            if files:
                cleared_count = 0
                for f in files:
                    os.remove(f)
                    cleared_count += 1
                logger.info(f"Cleared {cleared_count} models from {model_dir}")

        if os.path.exists("command_config.json"):
            os.remove("command_config.json")
            logger.info("Removed command_config.json")

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
        parent1, parent2 = select_mates(
            self.population,
            strategy=self.config_manager.mate_selection_strategy,
            dataset_name=self.config_manager.dataset_name,
            subset_percentage=self.config_manager.subset_percentage,
            seed=self.config_manager.seed
        )

        if parent1 and parent2:
            child = merge(
                parent1, parent2,
                strategy=self.config_manager.merge_strategy,
                validation_loader=self.validation_loader,
                seed=self.config_manager.seed,
                dampening_factor=self.config_manager.dampening_factor
            )
            child = mutate(
                child,
                generation=generation,
                mutation_rate=self.config_manager.mutation_rate,
                initial_mutation_strength=self.config_manager.initial_mutation_strength,
                decay_factor=self.config_manager.mutation_decay_factor
            )
            finetune(
                child,
                dataset_name=self.config_manager.dataset_name,
                validation_loader=self.validation_loader,
                epochs=self.config_manager.finetune_epochs,
                precision=self.config_manager.precision_config,
                seed=self.config_manager.seed,
                learning_rate=self.config_manager.learning_rate,
                scheduler_patience=self.config_manager.scheduler_patience,
                scheduler_factor=self.config_manager.scheduler_factor,
                subset_percentage=self.config_manager.subset_percentage,
                show_progress_bar=self.config_manager.show_progress_bar
            )
            self.population = create_next_generation(
                self.population,
                child,
                self.config_manager.population_size,
                dataset_name=self.config_manager.dataset_name,
                strategy=self.config_manager.generation_strategy,
                seed=self.config_manager.seed
            )
        else:
            logger.info("Population will carry over to the next generation without changes.")

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

    def _save_final_population(self) -> None:
        """Saves the final population of models."""
        logger.info("\n--- Saving final population to pretrained_models/ ---")
        model_dir = "src/pretrained_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if self.config_manager.delete_old_models:
            logger.info(f"Clearing old models loaded at the start of the run from {model_dir}...")
            # Also clear any files that match the pattern, in case they were not loaded
            pattern = os.path.join(model_dir, "model_niche_*.pth")
            old_model_files = glob.glob(pattern)
            files_to_delete = set(self.loaded_model_files + old_model_files)

            if not files_to_delete:
                logger.info("No old models found to clear.")
            else:
                for f in files_to_delete:
                    try:
                        if os.path.exists(f):
                            os.remove(f)
                            logger.info(f"  - Removed old model: {os.path.basename(f)}")
                    except OSError as e:
                        logger.error(f"Error removing file {f}: {e}")
        else:
            logger.info("`delete_old_models` is false. Skipping cleanup of old models.")

        for model_wrapper in self.population:
            niche_str = "_".join(map(str, model_wrapper.niche_classes))
            model_path = os.path.join(model_dir, f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth")
            torch.save(model_wrapper.model.state_dict(), model_path)
            logger.info(f"  - Saved model to {model_path}")
