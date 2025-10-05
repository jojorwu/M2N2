"""
This module contains the EvolutionSimulator class, which encapsulates the
entire logic for running an M2N2-inspired evolutionary experiment.
"""
import torch
import os
import glob
import re
import yaml
import numpy as np
import logging
import json
from .logger_config import setup_logger
from .model_wrapper import ModelWrapper
from .evolution import specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation
from .data import get_dataloaders
from .visualization import plot_fitness_history
from .utils import set_seed
from .enums import ModelName, DatasetName
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader

logger = logging.getLogger("M2N2_SIMULATOR")

class EvolutionSimulator:
    """
    Encapsulates the entire evolutionary simulation, from configuration
    loading to running the generational loop and saving the results.
    """
    config: Dict[str, Any]
    model_config: ModelName
    dataset_name: DatasetName
    precision_config: str
    num_generations: int
    population_size: int
    merge_strategy: str
    dampening_factor: float
    mutation_rate: float
    initial_mutation_strength: float
    mutation_decay_factor: float
    learning_rate: float
    scheduler_patience: int
    scheduler_factor: float
    subset_percentage: float
    validation_split: float
    batch_size: int
    specialize_epochs: int
    finetune_epochs: int
    device: torch.device
    seed: int
    population: List[ModelWrapper]
    fitness_history: List[Tuple[float, float]]
    validation_loader: DataLoader
    current_generation: int
    num_classes: int

    def __init__(self, config_path: str = 'config.yaml') -> None:
        """
        Initializes the simulator by loading configuration and setting up
        the environment.

        Args:
            config_path (str, optional): The path to the configuration YAML
                file. Defaults to 'config.yaml'.
        """
        self._setup_environment(config_path)
        self._initialize_parameters()

        logger.info("--- M2N2 Simplified Implementation ---")
        logger.info(f"Loaded configuration for model: {self.model_config}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Experiment seed: {self.seed}\n")

        # --- 3. Initialize Population and DataLoaders ---
        self.population = []
        self.fitness_history = []
        self.current_generation = 0
        self._initialize_dataloaders()
        self._initialize_population()
        self._initialize_fitness_log()

    def _setup_environment(self, config_path: str) -> None:
        """Loads configuration and sets up the logger."""
        # --- 1. Load Configuration from YAML ---
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # --- 2. Configure Logging ---
        log_file = self.config.get('log_file')
        setup_logger(log_file=log_file)

    def _initialize_parameters(self) -> None:
        """Initializes simulator parameters from the config."""
        # --- General settings ---
        self.model_config = ModelName(self.config['model_config'])
        self.dataset_name = DatasetName(self.config['dataset_name'])
        self.precision_config = str(self.config['precision_config'])
        self.num_generations = self.config['num_generations']
        self.population_size = self.config['population_size']

        # --- Evolutionary settings ---
        self.merge_strategy = self.config['merge_strategy']
        self.dampening_factor = self.config['fitness_weighted_merge_dampening_factor']
        self.mutation_rate = self.config['mutation_rate']
        self.initial_mutation_strength = self.config['initial_mutation_strength']
        self.mutation_decay_factor = self.config['mutation_decay_factor']

        # --- Optimizer settings ---
        self.learning_rate = self.config['optimizer_config']['learning_rate']

        # --- Scheduler settings ---
        self.scheduler_patience = self.config['scheduler_config']['patience']
        self.scheduler_factor = self.config['scheduler_config']['factor']

        # --- Data settings ---
        self.subset_percentage = self.config['subset_percentage']
        self.validation_split = self.config['validation_split']
        self.batch_size = self.config['batch_size']

        # --- Training epochs ---
        if self.model_config in self.config.get('model_specific_epochs', {}):
            self.specialize_epochs = self.config['model_specific_epochs'][self.model_config]['specialize']
            self.finetune_epochs = self.config['model_specific_epochs'][self.model_config]['finetune']
        else:
            self.specialize_epochs = self.config['default_epochs']['specialize']
            self.finetune_epochs = self.config['default_epochs']['finetune']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- Seed for reproducibility ---
        # Use seed from config if provided, otherwise generate a random one.
        self.seed = self.config.get('seed') or np.random.randint(0, 1_000_000)

    def _initialize_dataloaders(self) -> None:
        """Creates the necessary DataLoaders for the experiment."""
        logger.info("--- Creating DataLoaders ---")
        _, self.validation_loader, _, self.num_classes = get_dataloaders(
            dataset_name=self.dataset_name,
            model_name=self.model_config,
            batch_size=self.batch_size,
            subset_percentage=self.subset_percentage,
            validation_split=self.validation_split,
            seed=self.seed
        )

    def _initialize_population(self) -> None:
        """Initializes or loads the population of models."""
        if self.seed is not None:
            set_seed(self.seed)
        logger.info("--- STEP 1: Initializing or Loading Population ---")
        model_dir = "src/pretrained_models"
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if model_files:
            self._load_population_from_files(model_files)
        else:
            self._initialize_new_population()

    def _load_population_from_files(self, model_files: List[str]) -> None:
        """Loads a population of models from saved .pth files."""
        logger.info(f"Found {len(model_files)} models in 'src/pretrained_models'. Loading them.")
        for f in model_files:
            match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(f))
            if match:
                niche_classes = [int(n) for n in match.group(1).split('_')]
                fitness = float(match.group(2))
                wrapper = ModelWrapper(model_name=self.model_config, niche_classes=niche_classes, device=self.device, num_classes=self.num_classes)
                wrapper.model.load_state_dict(torch.load(f, map_location=self.device))
                wrapper.fitness = fitness
                # Ensure that loaded models are re-evaluated.
                wrapper.fitness_is_current = False
                self.population.append(wrapper)

    def _initialize_new_population(self) -> None:
        """Initializes a new population from scratch and specializes them."""
        logger.info("No pretrained models found. Initializing a new population from scratch.")
        niches = [[i] for i in range(self.population_size)]
        for i in range(self.population_size):
            self.population.append(ModelWrapper(model_name=self.model_config, niche_classes=niches[i], device=self.device, num_classes=self.num_classes))

        logger.info("--- Specializing Initial Models ---")
        for model_wrapper in self.population:
            specialize(
                model_wrapper,
                dataset_name=self.dataset_name,
                epochs=self.specialize_epochs,
                precision=self.precision_config,
                seed=self.seed,
                learning_rate=self.learning_rate,
                subset_percentage=self.subset_percentage
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
                        dataset_name=self.dataset_name,
                        epochs=self.specialize_epochs,
                        precision=self.precision_config,
                        seed=self.seed,
                        learning_rate=self.learning_rate,
                        subset_percentage=self.subset_percentage
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
            if not model_wrapper.fitness_is_current:
                evaluate(model_wrapper, dataset_name=self.dataset_name, subset_percentage=self.subset_percentage, seed=self.seed)

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
            files = glob.glob(os.path.join(model_dir, "*.pth"))
            if files:
                for f in files:
                    os.remove(f)
                logger.info(f"Cleared {len(files)} models from {model_dir}")

        if os.path.exists("command_config.json"):
            os.remove("command_config.json")
            logger.info("Removed command_config.json")

    def _restart(self) -> None:
        """Resets the simulation to its initial state for a fresh run."""
        logger.info("\n--- RESTARTING SIMULATION ---")
        self._clear_simulation_artifacts()

        # Reset state variables
        self.population = []
        self.fitness_history = []
        self.current_generation = 0

        # Re-initialize
        self._initialize_population()
        self._initialize_fitness_log()
        logger.info("--- Simulation has been restarted ---")


    def _load_dynamic_config(self) -> Dict[str, Any]:
        """
        Checks for and loads dynamic configuration from command_config.json,
        allowing for real-time control over the simulation.

        Returns:
            Dict[str, Any]: The command configuration.
        """
        config_path = 'command_config.json'
        if not os.path.exists(config_path):
            return {}

        try:
            with open(config_path, 'r') as f:
                command_config = json.load(f)

            # --- Update parameters if they exist in the command config ---
            updates = {
                'num_generations': (int, 'Number of Generations'),
                'population_size': (int, 'Population Size'),
                'mutation_rate': (float, 'Mutation Rate'),
                'merge_strategy': (str, 'Merge Strategy'),
                'initial_mutation_strength': (float, 'Initial Mutation Strength'),
                'mutation_decay_factor': (float, 'Mutation Decay Factor')
            }

            for key, (cast, name) in updates.items():
                new_value = command_config.get(key)
                if new_value is not None and new_value != getattr(self, key):
                    setattr(self, key, cast(new_value))
                    logger.info(f"Dynamically updated {name} to: {getattr(self, key)}")

            # Nested dictionaries for optimizer and scheduler
            if 'optimizer_config' in command_config:
                new_lr = command_config['optimizer_config'].get('learning_rate')
                if new_lr is not None and new_lr != self.learning_rate:
                    self.learning_rate = new_lr
                    logger.info(f"Dynamically updated Learning Rate to: {self.learning_rate}")

            if 'scheduler_config' in command_config:
                new_patience = command_config['scheduler_config'].get('patience')
                if new_patience is not None and new_patience != self.scheduler_patience:
                    self.scheduler_patience = new_patience
                    logger.info(f"Dynamically updated Scheduler Patience to: {self.scheduler_patience}")

                new_factor = command_config['scheduler_config'].get('factor')
                if new_factor is not None and new_factor != self.scheduler_factor:
                    self.scheduler_factor = new_factor
                    logger.info(f"Dynamically updated Scheduler Factor to: {self.scheduler_factor}")

            return command_config

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load or parse command config file: {e}")
            return {}

    def _run_evolution_phase(self, generation: int) -> None:
        """Handles the mating, mutation, and selection of models."""
        logger.info("--- Mating and Evolution ---")
        parent1, parent2 = select_mates(self.population, dataset_name=self.dataset_name, subset_percentage=self.subset_percentage, seed=self.seed)

        if parent1 and parent2:
            # Crossover
            child = merge(
                parent1, parent2,
                strategy=self.merge_strategy,
                validation_loader=self.validation_loader,
                seed=self.seed,
                dampening_factor=self.dampening_factor
            )
            # Mutation
            child = mutate(
                child,
                generation=generation,
                mutation_rate=self.mutation_rate,
                initial_mutation_strength=self.initial_mutation_strength,
                decay_factor=self.mutation_decay_factor
            )
            # Fine-tuning
            finetune(
                child,
                dataset_name=self.dataset_name,
                validation_loader=self.validation_loader,
                epochs=self.finetune_epochs,
                precision=self.precision_config,
                seed=self.seed,
                learning_rate=self.learning_rate,
                scheduler_patience=self.scheduler_patience,
                scheduler_factor=self.scheduler_factor,
                subset_percentage=self.subset_percentage
            )
            # Selection
            self.population = create_next_generation(
                self.population,
                child,
                self.population_size,
                dataset_name=self.dataset_name,
                seed=self.seed
            )
        else:
            logger.info("Population will carry over to the next generation without changes.")

    def run_one_generation(self) -> None:
        """
        Runs a single generation of the evolutionary simulation.
        """
        logger.info(f"\n--- GENERATION {self.current_generation + 1}/{self.num_generations} ---")
        self._run_specialization_phase(self.current_generation)
        self._run_evaluation_phase()
        self._run_evolution_phase(self.current_generation)
        self.current_generation += 1

    def run(self) -> None:
        """
        Runs the main evolutionary loop, checking for commands each generation.
        """
        while self.current_generation < self.num_generations:
            command_config = self._load_dynamic_config()

            if command_config.get('restart_simulation'):
                self._restart()
                continue

            if command_config.get('stop_simulation'):
                logger.info("Stop command received. Shutting down gracefully.")
                break

            self.run_one_generation()

        if self.fitness_history:
             self._summarize_and_save()
        else:
             logger.info("Simulation stopped before any generations were completed. No summary to generate.")

    def _summarize_and_save(self) -> None:
        """Prints a final summary, visualizes fitness history, and saves the final population."""
        logger.info("\n\n--- EXPERIMENT SUMMARY ---")
        logger.info("Fitness history (Best, Average):")
        for i, (best, avg) in enumerate(self.fitness_history):
            logger.info(f"  - Generation {i+1}: Best={best:.2f}%, Avg={avg:.2f}%")

        final_best_model = max(self.population, key=lambda m: m.fitness)
        logger.info(f"\nFinal best model achieved an accuracy of {final_best_model.fitness:.2f}%")

        # Visualize and Save
        plot_fitness_history(self.fitness_history, 'fitness_history.png')
        self._save_final_population()

    def _save_final_population(self) -> None:
        """Saves the models in the final population to the 'src/pretrained_models' directory."""
        logger.info("\n--- Saving final population to pretrained_models/ ---")
        model_dir = "src/pretrained_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            # Clear out old models from previous runs to prevent confusion.
            files = glob.glob(os.path.join(model_dir, "*.pth"))
            for f in files:
                os.remove(f)

        for i, model_wrapper in enumerate(self.population):
            niche_str = "_".join(map(str, model_wrapper.niche_classes))
            model_path = os.path.join(model_dir, f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth")
            torch.save(model_wrapper.model.state_dict(), model_path)
            logger.info(f"  - Saved model to {model_path}")