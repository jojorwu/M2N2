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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from .logger_config import setup_logger
from .utils import set_seed
from .evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation, _get_fitness_score
from .data import get_dataloaders
from .visualization import plot_fitness_history

logger = logging.getLogger("M2N2_SIMULATOR")

class EvolutionSimulator:
    """
    Encapsulates the entire evolutionary simulation, from configuration
    loading to running the generational loop and saving the results.
    """
    def __init__(self, config_path='config.yaml', seed=None):
        """
        Initializes the simulator by loading configuration and setting up
        the environment.

        Args:
            config_path (str, optional): The path to the configuration YAML
                file. Defaults to 'config.yaml'.
            seed (int, optional): A specific seed for reproducibility. If
                None, a random seed will be generated. Defaults to None.
        """
        self._setup_environment(config_path)
        self._initialize_parameters(seed=seed)

        logger.info("--- M2N2 Simplified Implementation ---")
        logger.info(f"Loaded configuration for model: {self.model_config}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Experiment seed: {self.seed}\n")

        # --- 3. Initialize Population and DataLoaders ---
        self.population = []
        self.fitness_history = []
        self._initialize_dataloaders()
        self._initialize_population()

    def _setup_environment(self, config_path):
        """Loads configuration and sets up the logger."""
        # --- 1. Load Configuration from YAML ---
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # --- 2. Configure Logging ---
        log_file = self.config.get('log_file')
        setup_logger(log_file=log_file)

    def _initialize_parameters(self, seed=None):
        """Initializes simulator parameters from the config."""
        # General settings
        self.model_config = self.config['model_config']
        self.dataset_name = self.config['dataset_name']
        self.model_dir = self.config['model_dir']
        self.precision_config = str(self.config['precision_config'])
        self.num_generations = self.config['num_generations']
        self.population_size = self.config['population_size']

        # Evolutionary settings
        self.merge_strategy = self.config['merge_strategy']
        self.mutation_rate = self.config['mutation_rate']
        self.initial_mutation_strength = self.config['initial_mutation_strength']
        self.mutation_decay_factor = self.config['mutation_decay_factor']

        # Data settings
        self.subset_percentage = self.config['subset_percentage']
        self.validation_split = self.config['validation_split']

        # Determine epochs based on model-specific or default settings
        if self.model_config in self.config.get('model_specific_epochs', {}):
            self.specialize_epochs = self.config['model_specific_epochs'][self.model_config]['specialize']
            self.finetune_epochs = self.config['model_specific_epochs'][self.model_config]['finetune']
        else:
            self.specialize_epochs = self.config['default_epochs']['specialize']
            self.finetune_epochs = self.config['default_epochs']['finetune']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 1_000_000)
        set_seed(self.seed)

    def _initialize_dataloaders(self):
        """Creates the necessary DataLoaders for the experiment."""
        logger.info("--- Creating DataLoaders ---")
        _, self.validation_loader, _ = get_dataloaders(
            dataset_name=self.dataset_name,
            model_name=self.model_config,
            batch_size=64,
            subset_percentage=self.subset_percentage,
            validation_split=self.validation_split,
            seed=self.seed
        )

    def _initialize_population(self):
        """Initializes or loads the population of models."""
        logger.info("--- STEP 1: Initializing or Loading Population ---")
        niches = [[i] for i in range(self.population_size)]
        model_files = glob.glob(os.path.join(self.model_dir, "*.pth"))

        if model_files:
            logger.info(f"Found {len(model_files)} models in {self.model_dir}. Loading them.")
            for f in model_files:
                match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(f))
                if match:
                    niche_classes = [int(n) for n in match.group(1).split('_')]
                    fitness = float(match.group(2))
                    wrapper = ModelWrapper(model_name=self.model_config, niche_classes=niche_classes, device=self.device)
                    wrapper.model.load_state_dict(torch.load(f, map_location=self.device))
                    wrapper.fitness = fitness
                    # BUG FIX: Ensure that loaded models are re-evaluated.
                    # Their fitness from a previous run is not guaranteed to be
                    # relevant for the current run's (potentially different)
                    # dataset subset.
                    wrapper.fitness_is_current = False
                    self.population.append(wrapper)
        else:
            logger.info("No pretrained models found. Initializing a new population from scratch.")
            initial_population = [ModelWrapper(model_name=self.model_config, niche_classes=niches[i], device=self.device) for i in range(self.population_size)]

            logger.info("--- Specializing Initial Models (in parallel) ---")
            specialize_fn = partial(specialize, dataset_name=self.dataset_name, epochs=self.specialize_epochs, precision=self.precision_config, seed=self.seed)
            with ProcessPoolExecutor() as executor:
                self.population = list(executor.map(specialize_fn, initial_population))
            logger.info("")

    def run(self):
        """Runs the main evolutionary loop."""
        for generation in range(self.num_generations):
            logger.info(f"\n--- GENERATION {generation + 1}/{self.num_generations} ---")

            if generation > 0:
                logger.info("--- Specializing Models ---")
                for model_wrapper in self.population:
                    if model_wrapper.niche_classes != list(range(10)):
                        specialize(model_wrapper, dataset_name=self.dataset_name, epochs=self.specialize_epochs, precision=self.precision_config, seed=self.seed)
                logger.info("")

            logger.info("--- Evaluating Population on Test Set (in parallel) ---")
            # Identify models that need evaluation
            models_to_evaluate = [m for m in self.population if not m.fitness_is_current]

            if models_to_evaluate:
                # Use a partial function to pass static arguments to the mapping function
                eval_fn = partial(_get_fitness_score, dataset_name=self.dataset_name, seed=self.seed)

                with ProcessPoolExecutor() as executor:
                    # Map the evaluation function to the models that need it
                    fitness_scores = list(executor.map(eval_fn, models_to_evaluate))

                # Update fitness scores on the original population objects
                for i, model_wrapper in enumerate(models_to_evaluate):
                    model_wrapper.fitness = fitness_scores[i]
                    model_wrapper.fitness_is_current = True

            # Log if any models were skipped
            num_skipped = len(self.population) - len(models_to_evaluate)
            if num_skipped > 0:
                logger.info(f"Skipped evaluation for {num_skipped} model(s) with up-to-date fitness.")

            best_fitness = max([m.fitness for m in self.population])
            avg_fitness = sum([m.fitness for m in self.population]) / len(self.population)
            self.fitness_history.append((best_fitness, avg_fitness))
            logger.info(f"\nGeneration {generation + 1} Stats: Best Fitness = {best_fitness:.2f}%, Avg Fitness = {avg_fitness:.2f}%\n")

            logger.info("--- Mating and Evolution ---")
            parent1, parent2 = select_mates(self.population, dataset_name=self.dataset_name, seed=self.seed)

            if parent1 and parent2:
                child = merge(parent1, parent2, strategy=self.merge_strategy, validation_loader=self.validation_loader, seed=self.seed)
                child = mutate(
                    child,
                    generation=generation,
                    mutation_rate=self.mutation_rate,
                    initial_mutation_strength=self.initial_mutation_strength,
                    decay_factor=self.mutation_decay_factor
                )
                finetune(child, dataset_name=self.dataset_name, epochs=self.finetune_epochs, precision=self.precision_config, seed=self.seed)
                self.population = create_next_generation(self.population, child, self.population_size, dataset_name=self.dataset_name, seed=self.seed)
            else:
                logger.info("Population will carry over to the next generation without changes.")

        self._summarize_and_save()

    def _summarize_and_save(self):
        """Prints a final summary and saves the results."""
        logger.info("\n\n--- EXPERIMENT SUMMARY ---")
        logger.info("Fitness history (Best, Average):")
        for i, (best, avg) in enumerate(self.fitness_history):
            logger.info(f"  - Generation {i+1}: Best={best:.2f}%, Avg={avg:.2f}%")

        final_best_model = max(self.population, key=lambda m: m.fitness)
        logger.info(f"\nFinal best model achieved an accuracy of {final_best_model.fitness:.2f}%")

        # Visualize and Save
        plot_fitness_history(self.fitness_history, 'fitness_history.png')

        logger.info(f"\n--- Saving final population to {self.model_dir}/ ---")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            files = glob.glob(os.path.join(self.model_dir, "*.pth"))
            for f in files:
                os.remove(f)

        for i, model_wrapper in enumerate(self.population):
            niche_str = "_".join(map(str, model_wrapper.niche_classes))
            model_path = os.path.join(self.model_dir, f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth")
            torch.save(model_wrapper.model.state_dict(), model_path)
            logger.info(f"  - Saved model to {model_path}")