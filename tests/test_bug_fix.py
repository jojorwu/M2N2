import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import yaml
import sys

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model_wrapper import ModelWrapper
from src.enums import ModelName

class TestBugFix(unittest.TestCase):
    """
    Contains the specific test case to verify the bug fix for the hardcoded
    number of classes in the specialization phase.
    """

    def setUp(self):
        """Set up a temporary environment for the test."""
        self.test_dir = "tests/temp_bug_fix_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)

        # A base config that uses a non-standard number of classes
        self.base_config = {
            'model_config': 'CIFAR10', # Using a standard model for simplicity
            'dataset_name': 'CIFAR10', # This will be mocked to have 5 classes
            'precision_config': '32',
            'num_generations': 2,
            'population_size': 1,
            'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0, # Added missing key
            'mutation_rate': 0.05,
            'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 0.9,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 64,
            'subset_percentage': 0.01,
            'validation_split': 0.1,
            'default_epochs': {'specialize': 1, 'finetune': 1},
            'log_file': None,
        }

    def tearDown(self):
        """Clean up the temporary environment after the test."""
        shutil.rmtree(self.test_dir)

    @patch('src.simulator.specialize')
    @patch('src.simulator.get_dataloaders')
    @patch('src.simulator.glob.glob')
    def test_specialization_is_skipped_for_generalist_with_custom_num_classes(self, mock_glob, mock_get_dataloaders, mock_specialize):
        """
        Tests that the specialization phase correctly identifies a generalist
        model and skips specializing it, even when the number of classes is
        not the default of 10. This test verifies the fix for the hardcoded
        `range(10)` bug.
        """
        # Arrange
        # 1. Prevent loading of pretrained models from other tests
        mock_glob.return_value = []

        # 2. Mock get_dataloaders to return a custom number of classes (e.g., 5)
        custom_num_classes = 5
        # Create mock DataLoaders that are not None
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_get_dataloaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader, custom_num_classes)

        # 3. Write a config file for the test
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        # 4. Initialize the simulator. This will set self.num_classes to 5
        # and will call `specialize` on its newly created population during init.
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.num_classes, custom_num_classes, "Simulator did not correctly set the custom number of classes.")

        # 4a. Reset the mock to ignore the calls from the initialization phase.
        mock_specialize.reset_mock()

        # 5. Create a "generalist" model wrapper for this custom configuration.
        # Its niche covers all classes from 0 to 4.
        generalist_wrapper = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=list(range(custom_num_classes)),
            device='cpu',
            num_classes=custom_num_classes
        )
        simulator.population = [generalist_wrapper]

        # Act
        # Run the specialization phase for a generation > 0. This is where the
        # bug would have occurred.
        simulator._run_specialization_phase(generation=1)

        # Assert
        # The `specialize` function should NOT have been called, because the model
        # is a generalist. The fix ensures that the check `list(range(self.num_classes))`
        # correctly identifies it as such.
        mock_specialize.assert_not_called()

if __name__ == '__main__':
    unittest.main()