import unittest
import os
import yaml
import shutil
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_manager import ConfigManager

class TestConfigManagerValidation(unittest.TestCase):
    """
    Tests the validation logic within the ConfigManager to ensure it catches
    invalid configuration parameters.
    """

    def setUp(self):
        self.test_dir = "tests/temp_config_test"
        os.makedirs(self.test_dir, exist_ok=True)
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        self.base_config = {
            'model_name': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 2, # Valid default
            'mate_selection_strategy': 'healing',
            'generation_strategy': 'replace_worst',
            'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 64,
            'mutation_rate': 0.1, # Valid default
            'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 1.0,
            'subset_percentage': 0.1, # Valid default
            'validation_split': 0.1,
            'default_epochs': {'specialize': 1, 'finetune': 1},
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def write_config(self, config_data):
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)

    def test_valid_config_loads_successfully(self):
        """Tests that a valid configuration file loads without raising an error."""
        self.write_config(self.base_config)
        try:
            ConfigManager(config_path=self.config_path)
        except ValueError:
            self.fail("Valid config raised ValueError unexpectedly.")

    def test_invalid_mutation_rate_raises_error(self):
        """Tests that a mutation_rate outside the [0, 1] range raises a ValueError."""
        config = self.base_config.copy()
        config['mutation_rate'] = 1.1
        self.write_config(config)
        with self.assertRaisesRegex(ValueError, "mutation_rate must be between 0.0 and 1.0"):
            ConfigManager(config_path=self.config_path)

        config['mutation_rate'] = -0.1
        self.write_config(config)
        with self.assertRaisesRegex(ValueError, "mutation_rate must be between 0.0 and 1.0"):
            ConfigManager(config_path=self.config_path)

    def test_invalid_subset_percentage_raises_error(self):
        """Tests that a subset_percentage outside the [0, 1] range raises a ValueError."""
        config = self.base_config.copy()
        config['subset_percentage'] = 1.5
        self.write_config(config)
        with self.assertRaisesRegex(ValueError, "subset_percentage must be between 0.0 and 1.0"):
            ConfigManager(config_path=self.config_path)

    def test_negative_learning_rate_raises_error(self):
        """Tests that a negative learning_rate raises a ValueError."""
        config = self.base_config.copy()
        config['optimizer_config']['learning_rate'] = -0.01
        self.write_config(config)
        with self.assertRaisesRegex(ValueError, "learning_rate must be non-negative"):
            ConfigManager(config_path=self.config_path)

    def test_invalid_population_size_raises_error(self):
        """Tests that a population_size of 1 or less raises a ValueError."""
        config = self.base_config.copy()
        config['population_size'] = 1
        self.write_config(config)
        with self.assertRaisesRegex(ValueError, "population_size must be greater than 1"):
            ConfigManager(config_path=self.config_path)

if __name__ == '__main__':
    unittest.main()
