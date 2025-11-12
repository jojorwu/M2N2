import unittest
import os
import shutil
import yaml
from unittest.mock import patch, MagicMock

from src.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_config_test"
        os.makedirs(self.test_dir, exist_ok=True)
        self.config_path = os.path.join(self.test_dir, "config.yaml")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_missing_required_key_raises_value_error(self):
        """
        Ensures that if a required key is missing, a ValueError is raised.
        """
        base_config = {
            'model_name': 'CNN', 'dataset_name': 'CIFAR10', 'precision_config': '32',
            'num_generations': 1, 'population_size': 2, 'num_offspring': 1,
            'mate_selection_strategy': 'healing', 'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'mutation_rate': 0.1, 'initial_mutation_strength': 0.1, 'mutation_decay_factor': 1.0,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'subset_percentage': 0.01, 'validation_split': 0.1, 'batch_size': 4,
            'default_epochs': {'specialize': 1, 'finetune': 1}
        }

        required_keys = list(base_config.keys())

        for key in required_keys:
            with self.subTest(missing_key=key):
                config = base_config.copy()
                del config[key]
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f)

                with self.assertRaises(ValueError):
                    ConfigManager(config_path=self.config_path)

    def test_constructor_handles_corrupted_yaml_file(self):
        """
        Ensures the ConfigManager constructor raises a ValueError for a
        syntactically incorrect YAML file.
        """
        with open(self.config_path, "w") as f:
            f.write("model_name: CNN\n: invalid_syntax")

        with self.assertRaisesRegex(ValueError, "Error parsing YAML file"):
            ConfigManager(config_path=self.config_path)

    def test_constructor_handles_empty_yaml_file(self):
        """
        Ensures the ConfigManager constructor raises a ValueError if the
        YAML file is empty or does not produce a dictionary.
        """
        with open(self.config_path, "w") as f:
            f.write("") # Empty file

        with self.assertRaisesRegex(ValueError, "is invalid; expected a dictionary"):
            ConfigManager(config_path=self.config_path)

        with open(self.config_path, "w") as f:
            f.write("just_a_string") # Not a dictionary

        with self.assertRaisesRegex(ValueError, "is invalid; expected a dictionary"):
            ConfigManager(config_path=self.config_path)

    def test_constructor_handles_file_not_found(self):
        """
        Ensures the ConfigManager constructor raises a ValueError if the
        config file does not exist.
        """
        non_existent_path = os.path.join(self.test_dir, "non_existent_config.yaml")
        with self.assertRaisesRegex(ValueError, "Configuration file not found"):
            ConfigManager(config_path=non_existent_path)

if __name__ == '__main__':
    unittest.main()
