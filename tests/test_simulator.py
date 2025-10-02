import unittest
from unittest.mock import patch
import os
import shutil
import unittest
from unittest.mock import patch
import os
import shutil
import torch
import yaml
import sys
import json

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model import CifarCNN

class TestSimulatorInitialization(unittest.TestCase):
    """Unit tests for the EvolutionSimulator's initialization logic."""

    def setUp(self):
        """Set up a temporary environment for the simulator test."""
        self.test_dir = "tests/temp_simulator_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)

        # Base config that can be modified by each test
        self.base_config = {
            'model_config': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 1,
            'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 64,
            'mutation_rate': 0.0,
            'initial_mutation_strength': 0.0,
            'mutation_decay_factor': 1.0,
            'subset_percentage': 0.01,
            'validation_split': 0.1,
            'default_epochs': {'specialize': 0, 'finetune': 0},
        }

    def tearDown(self):
        """Clean up the temporary environment after the test."""
        shutil.rmtree(self.test_dir)
        if os.path.exists("command_config.json"):
            os.remove("command_config.json")

    @patch('src.simulator.glob.glob')
    def test_loaded_model_fitness_is_marked_as_stale(self, mock_glob):
        """
        Tests that a model loaded from a file has its `fitness_is_current`
        flag set to False, forcing a re-evaluation on the new dataset.
        """
        # Arrange
        model_dir = os.path.join(self.test_dir, "pretrained_models")
        os.makedirs(model_dir, exist_ok=True)
        dummy_model_path = os.path.join(model_dir, "model_niche_0_fitness_99.9.pth")
        torch.save(CifarCNN().state_dict(), dummy_model_path)

        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        mock_glob.return_value = [dummy_model_path]

        # Act
        simulator = EvolutionSimulator(config_path=self.config_path)

        # Assert
        self.assertGreater(len(simulator.population), 0, "Simulator failed to load any models.")
        self.assertFalse(simulator.population[0].fitness_is_current)

    def test_logging_is_configurable(self):
        """
        Tests that file logging can be enabled or disabled via the config file.
        """
        # --- Part 1: Test that the log file IS created when specified ---
        log_file_path = os.path.join(self.test_dir, "test.log")
        config_with_log = self.base_config.copy()
        config_with_log['log_file'] = log_file_path
        with open(self.config_path, 'w') as f:
            yaml.dump(config_with_log, f)

        # Act
        EvolutionSimulator(config_path=self.config_path)

        # Assert
        self.assertTrue(os.path.exists(log_file_path), "Log file was not created when path was specified.")

        # Clean up the created log file before the next assertion
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        # --- Part 2: Test that the log file IS NOT created when disabled ---
        config_without_log = self.base_config.copy()
        config_without_log['log_file'] = None # Explicitly disable
        with open(self.config_path, 'w') as f:
            yaml.dump(config_without_log, f)

        # Act
        EvolutionSimulator(config_path=self.config_path)

        # Assert
        # Check that the specific log file was not created.
        self.assertFalse(os.path.exists(log_file_path), "Log file was created even when disabled in config.")

    def test_simulator_uses_seed_from_config(self):
        """
        Tests that the simulator correctly uses the seed provided in the
        config file for reproducible experiments.
        """
        # --- Part 1: Test that a specified seed is used ---
        config_with_seed = self.base_config.copy()
        expected_seed = 12345
        config_with_seed['seed'] = expected_seed
        with open(self.config_path, 'w') as f:
            yaml.dump(config_with_seed, f)

        # Act
        simulator = EvolutionSimulator(config_path=self.config_path)

        # Assert
        self.assertEqual(simulator.seed, expected_seed,
                         "Simulator did not use the seed from the config file.")

    @patch('numpy.random.randint', return_value=54321)
    def test_simulator_generates_random_seed_if_not_provided(self, mock_randint):
        """
        Tests that the simulator generates a random seed when one is not
        specified in the config file.
        """
        # Arrange: Config without a 'seed' key
        config_without_seed = self.base_config.copy()
        # Ensure 'seed' key is not present
        if 'seed' in config_without_seed:
            del config_without_seed['seed']
        with open(self.config_path, 'w') as f:
            yaml.dump(config_without_seed, f)

        # Act
        simulator = EvolutionSimulator(config_path=self.config_path)

        # Assert
        # Check that the simulator's seed is the one from our mocked function
        self.assertEqual(simulator.seed, 54321,
                         "Simulator did not generate a random seed when none was provided.")
        mock_randint.assert_called_once()

    @patch('src.simulator.select_mates', return_value=(None, None))
    def test_simulator_loads_dynamic_config_from_command_file(self, mock_select_mates):
        """
        Tests that the simulator correctly loads and applies parameters
        from command_config.json during the evolution phase.
        """
        # Arrange
        # 1. Create a standard config file with default values
        config = self.base_config.copy()
        config['merge_strategy'] = 'average'
        config['mutation_rate'] = 0.1
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        # 2. Create a command config file with new values
        command_config = {
            "merge_strategy": "fitness_weighted",
            "mutation_rate": 0.99
        }
        with open("command_config.json", 'w') as f:
            json.dump(command_config, f)

        # 3. Instantiate the simulator and verify its initial state
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.merge_strategy, 'average')
        self.assertEqual(simulator.mutation_rate, 0.1)

        # Act
        # Call the evolution phase, which triggers the dynamic config load.
        # We patch select_mates to prevent the rest of the function from running.
        simulator._run_evolution_phase(generation=1)

        # Assert
        self.assertEqual(simulator.merge_strategy, 'fitness_weighted',
                         "Merge strategy was not dynamically updated.")
        self.assertEqual(simulator.mutation_rate, 0.99,
                         "Mutation rate was not dynamically updated.")
        mock_select_mates.assert_called_once()


if __name__ == '__main__':
    unittest.main()