import unittest
from unittest.mock import patch
import os
import shutil
import torch
import yaml
import sys

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2n2_implementation.simulator import EvolutionSimulator
from m2n2_implementation.model import CifarCNN

class TestSimulatorInitialization(unittest.TestCase):
    """Unit tests for the EvolutionSimulator's initialization logic."""

    def setUp(self):
        """Set up a temporary environment for the simulator test."""
        self.test_dir = "tests/temp_simulator_test"
        self.model_dir = os.path.join(self.test_dir, "pretrained_models")
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")

        # Create temporary directories
        os.makedirs(self.model_dir, exist_ok=True)

        # Create a dummy model state_dict to be "loaded"
        dummy_model = CifarCNN()
        self.model_path = os.path.join(self.model_dir, "model_niche_0_fitness_99.9.pth")
        torch.save(dummy_model.state_dict(), self.model_path)

        # Create a dummy config file for the simulator
        # The parameters are minimal to ensure the simulator can initialize.
        self.config = {
            'model_config': 'CIFAR10',
            'dataset_name': 'CIFAR10',
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
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        """Clean up the temporary environment after the test."""
        shutil.rmtree(self.test_dir)

    @patch('m2n2_implementation.simulator.glob.glob')
    def test_loaded_model_fitness_is_marked_as_stale(self, mock_glob):
        """
        Tests that a model loaded from a file has its `fitness_is_current`
        flag set to False, forcing a re-evaluation on the new dataset.

        This test is designed to FAIL with the original implementation, which
        incorrectly sets the flag to True, leading to the use of stale data.
        """
        # Arrange:
        # Make the mocked glob.glob return the path to our temporary model file
        # when the simulator tries to find models in its hardcoded directory.
        mock_glob.return_value = [self.model_path]

        # Act:
        # Instantiate the simulator. The __init__ method calls
        # `_initialize_population`, which will use our mocked glob.
        simulator = EvolutionSimulator(config_path=self.config_path)

        # Assert:
        # Check that the loaded model is correctly marked for re-evaluation.
        self.assertGreater(len(simulator.population), 0, "Simulator failed to load any models.")

        loaded_model_wrapper = simulator.population[0]

        # This is the key assertion. With the bug, this flag is True.
        # The fix is to ensure it is False.
        self.assertFalse(
            loaded_model_wrapper.fitness_is_current,
            "Loaded model's fitness_is_current flag was incorrectly set to True, "
            "which will cause the simulation to use stale fitness data from the "
            "previous experiment."
        )

if __name__ == '__main__':
    unittest.main()