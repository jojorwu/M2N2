import unittest
from unittest.mock import patch
import os
import shutil
import torch
import yaml
import sys

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
            'model_dir': os.path.join(self.test_dir, 'models'),
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 1,
            'merge_strategy': 'average',
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

    def test_initial_specialization_is_deterministic_with_seed(self):
        """
        Tests that when a seed is provided, two simulators created with the
        same config will have identical initial populations after specialization.
        This test is designed to FAIL until the bug is fixed.
        """
        # Arrange: Config with non-zero specialization epochs
        config = self.base_config.copy()
        config['default_epochs']['specialize'] = 1  # Enable specialization
        config['population_size'] = 2
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        # Act
        # Create two simulators with the same fixed seed. The `__init__` method
        # now handles setting the seed and initializing the population.
        sim1 = EvolutionSimulator(config_path=self.config_path, seed=42)
        model1_params = [p.clone() for p in sim1.population[0].model.parameters()]

        # Re-create the simulator with the same seed to ensure a fresh but
        # identical initialization.
        sim2 = EvolutionSimulator(config_path=self.config_path, seed=42)
        model2_params = [p.clone() for p in sim2.population[0].model.parameters()]

        # Assert
        # The weights should be identical if the seed was used for specialization
        self.assertEqual(len(model1_params), len(model2_params))
        for p1, p2 in zip(model1_params, model2_params):
            self.assertTrue(torch.equal(p1, p2),
                            "Model weights are not identical between two simulators with the same seed. "
                            "Initial specialization is likely not using the provided seed.")

if __name__ == '__main__':
    unittest.main()