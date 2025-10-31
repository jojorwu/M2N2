import unittest
import os
import shutil
import yaml
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_config_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)
        self.base_config = {
            'model_name': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 2,
            'mate_selection_strategy': 'healing',
            'generation_strategy': 'replace_worst',
            'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 64,
            'mutation_rate': 0.1,
            'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 1.0,
            'subset_percentage': 0.1,
            'validation_split': 0.1,
            'default_epochs': {'specialize': 1, 'finetune': 1},
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_missing_required_key_raises_value_error(self):
        """
        Tests that initializing with a config missing a required key
        raises a ValueError with an informative message.
        """
        incomplete_config = self.base_config.copy()
        del incomplete_config['population_size'] # remove a required key

        with open(self.config_path, 'w') as f:
            yaml.dump(incomplete_config, f)

        with self.assertRaisesRegex(ValueError, "Missing required configuration key: 'population_size'"):
            ConfigManager(config_path=self.config_path)

    def test_missing_non_essential_key_uses_default(self):
        """
        Tests that a missing non-essential key (one with a default) is
        handled gracefully.
        """
        config_without_optional = self.base_config.copy()
        # 'delete_old_models' has a default value of True
        if 'delete_old_models' in config_without_optional:
            del config_without_optional['delete_old_models']

        with open(self.config_path, 'w') as f:
            yaml.dump(config_without_optional, f)

        try:
            cm = ConfigManager(config_path=self.config_path)
            self.assertTrue(cm.delete_old_models) # Check that the default was applied
        except ValueError:
            self.fail("ConfigManager raised ValueError on a missing non-essential key.")

if __name__ == '__main__':
    unittest.main()
