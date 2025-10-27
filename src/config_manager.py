"""
Manages the configuration for the M2N2 simulation.

This module defines the ConfigManager class, which is responsible for
loading the static YAML configuration, handling dynamic updates from a JSON
command file, and providing a centralized, validated source of parameters
for the entire simulation.
"""
import yaml
import json
import os
import logging
import numpy as np
from typing import Dict, Any
from .enums import ModelName, DatasetName

logger = logging.getLogger("M2N2_SIMULATOR")

class ConfigManager:
    """
    Handles all configuration loading and dynamic updates for the simulator.
    """
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initializes the ConfigManager by loading and parsing the base config.

        Args:
            config_path (str): The path to the main YAML configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initializes all simulation parameters from the loaded config."""
        # --- General settings ---
        self.model_config = ModelName(self.config['model_config'])
        self.dataset_name = DatasetName(self.config['dataset_name'])
        self.precision_config = str(self.config['precision_config'])
        self.num_generations = self.config['num_generations']
        self.population_size = self.config['population_size']
        self.delete_old_models = self.config.get('delete_old_models', True)

        # --- Evolutionary settings ---
        self.mate_selection_strategy = self.config['mate_selection_strategy']
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

        # --- UI settings ---
        self.show_progress_bar = self.config.get('show_progress_bar', True)

        # --- Training epochs ---
        if self.model_config in self.config.get('model_specific_epochs', {}):
            self.specialize_epochs = self.config['model_specific_epochs'][self.model_config]['specialize']
            self.finetune_epochs = self.config['model_specific_epochs'][self.model_config]['finetune']
        else:
            self.specialize_epochs = self.config['default_epochs']['specialize']
            self.finetune_epochs = self.config['default_epochs']['finetune']

        # --- Seed for reproducibility ---
        self.seed = self.config.get('seed') or np.random.randint(0, 1_000_000)

        # --- Configuration Validation ---
        if self.merge_strategy == 'sequential_constructive' and self.validation_split <= 0:
            raise ValueError("The 'sequential_constructive' merge strategy requires a validation_split > 0")

    def load_dynamic_config(self) -> Dict[str, Any]:
        """
        Checks for and applies dynamic configuration from command_config.json.
        """
        config_path = 'command_config.json'
        if not os.path.exists(config_path):
            return {}

        try:
            with open(config_path, 'r') as f:
                command_config = json.load(f)

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
