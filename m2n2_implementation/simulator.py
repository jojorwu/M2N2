"""
This module contains the EvolutionSimulator class, which encapsulates the
entire logic for running an M2N2-inspired evolutionary experiment.
"""
import torch
import os
import glob
import re
import yaml
from .logger_config import logger
from .evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation
from .data import get_dataloaders
from .visualization import plot_fitness_history

class EvolutionSimulator:
    """
    Encapsulates the entire evolutionary simulation, from configuration
    loading to running the generational loop and saving the results.
    """
    def __init__(self, config_path='config.yaml'):
        """
        Initializes the simulator by loading configuration and setting up
        the environment.

        Args:
            config_path (str, optional): The path to the configuration YAML
                file. Defaults to 'config.yaml'.
        """
        # --- 1. Load Configuration from YAML ---
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # General settings
        self.model_config = self.config['model_config']
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
        logger.info("--- M2N2 Simplified Implementation ---")
        logger.info(f"Loaded configuration for model: {self.model_config}")
        logger.info(f"Using device: {self.device}\n")

        # --- 2. Initialize Population and DataLoaders ---
        self.population = []
        self.fitness_history = []
        self._initialize_dataloaders()
        self._initialize_population()

    def _initialize_dataloaders(self):
        """Creates the necessary DataLoaders for the experiment."""
        logger.info("--- Creating DataLoaders ---")
        _, self.validation_loader, _ = get_dataloaders(
            dataset_name=self.model_config,
            batch_size=64,
            subset_percentage=self.subset_percentage,
            validation_split=self.validation_split
        )

    def _initialize_population(self):
        """Initializes or loads the population of models."""
        logger.info("--- STEP 1: Initializing or Loading Population ---")
        model_dir = "m2n2_implementation/pretrained_models"
        niches = [[i] for i in range(self.population_size)]
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if model_files:
            logger.info(f"Found {len(model_files)} models in {model_dir}. Loading them.")
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
            for i in range(self.population_size):
                self.population.append(ModelWrapper(model_name=self.model_config, niche_classes=niches[i], device=self.device))

            logger.info("--- Specializing Initial Models ---")
            for model_wrapper in self.population:
                specialize(model_wrapper, epochs=self.specialize_epochs, precision=self.precision_config)
            logger.info("")

    def run(self):
        """Runs the main evolutionary loop."""
        for generation in range(self.num_generations):
            logger.info(f"\n--- GENERATION {generation + 1}/{self.num_generations} ---")

            if generation > 0:
                logger.info("--- Specializing Models ---")
                for model_wrapper in self.population:
                    if model_wrapper.niche_classes != list(range(10)):
                        specialize(model_wrapper, epochs=self.specialize_epochs, precision=self.precision_config)
                logger.info("")

            logger.info("--- Evaluating Population on Test Set ---")
            for model_wrapper in self.population:
                evaluate(model_wrapper)

            best_fitness = max([m.fitness for m in self.population])
            avg_fitness = sum([m.fitness for m in self.population]) / len(self.population)
            self.fitness_history.append((best_fitness, avg_fitness))
            logger.info(f"\nGeneration {generation + 1} Stats: Best Fitness = {best_fitness:.2f}%, Avg Fitness = {avg_fitness:.2f}%\n")

            logger.info("--- Mating and Evolution ---")
            parent1, parent2 = select_mates(self.population)

            if parent1 and parent2:
                child = merge(parent1, parent2, strategy=self.merge_strategy, validation_loader=self.validation_loader)
                child = mutate(
                    child,
                    generation=generation,
                    mutation_rate=self.mutation_rate,
                    initial_mutation_strength=self.initial_mutation_strength,
                    decay_factor=self.mutation_decay_factor
                )
                finetune(child, epochs=self.finetune_epochs, precision=self.precision_config)
                self.population = create_next_generation(self.population, child, self.population_size)
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

        logger.info("\n--- Saving final population to pretrained_models/ ---")
        model_dir = "m2n2_implementation/pretrained_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            files = glob.glob(os.path.join(model_dir, "*.pth"))
            for f in files:
                os.remove(f)

        for i, model_wrapper in enumerate(self.population):
            niche_str = "_".join(map(str, model_wrapper.niche_classes))
            model_path = os.path.join(model_dir, f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth")
            torch.save(model_wrapper.model.state_dict(), model_path)
            logger.info(f"  - Saved model to {model_path}")