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
        # Clean up model dir if it was created by a test
        model_dir = "src/pretrained_models"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    def test_save_final_population_preserves_unrelated_files(self):
        """
        Tests that _save_final_population only deletes the models loaded at the
        start of the run and preserves any other user-created files in the
        pretrained_models directory.
        """
        # Arrange
        # 1. Create the pretrained_models directory and dummy files
        model_dir = "src/pretrained_models"
        os.makedirs(model_dir, exist_ok=True)

        loaded_model_path = os.path.join(model_dir, "model_niche_0_fitness_10.00.pth")
        user_file_path = os.path.join(model_dir, "user_backup_model.pth")

        # Create a dummy model state dict to save
        dummy_model = CifarCNN()
        torch.save(dummy_model.state_dict(), loaded_model_path)
        # Create a simple user file
        with open(user_file_path, "w") as f:
            f.write("This is a user backup, do not delete.")

        # 2. Configure the simulator to run for one generation
        config = self.base_config.copy()
        config['num_generations'] = 1
        config['population_size'] = 1
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        # Act
        # Instantiate the simulator. It will load the dummy model.
        simulator = EvolutionSimulator(config_path=self.config_path)
        # Run the simulation. This will trigger _save_final_population at the end.
        simulator.run()

        # Assert
        # 3. Check that the user's file is still there
        self.assertTrue(os.path.exists(user_file_path),
                        "The user's unrelated file was deleted.")

        # 4. Check that the original loaded model file is gone
        self.assertFalse(os.path.exists(loaded_model_path),
                         "The original loaded model file was not deleted.")

        # 5. Check that at least one new model was saved
        new_model_files = [f for f in os.listdir(model_dir) if f.startswith('model_niche_')]
        self.assertGreater(len(new_model_files), 0,
                           "No new model was saved to the directory.")

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

    @patch('src.simulator.EvolutionSimulator._run_specialization_phase')
    @patch('src.simulator.EvolutionSimulator._run_evaluation_phase')
    @patch('src.simulator.select_mates', return_value=(None, None))
    def test_simulator_loads_dynamic_config_from_command_file(self, mock_select_mates, mock_evaluation, mock_specialization):
        """
        Tests that the simulator correctly loads and applies parameters
        from command_config.json during a generation run.
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
        # Run the simulation, which should trigger the dynamic config load
        # before the first generation. We patch the other phases to isolate
        # the config loading and evolution phase.
        simulator.run()

        # Assert
        self.assertEqual(simulator.merge_strategy, 'fitness_weighted',
                         "Merge strategy was not dynamically updated.")
        self.assertEqual(simulator.mutation_rate, 0.99,
                         "Mutation rate was not dynamically updated.")
        mock_select_mates.assert_called_once()

    @patch('src.simulator.specialize')
    @patch('src.simulator.get_dataloaders')
    @patch('src.simulator.glob.glob')
    def test_specialization_is_skipped_for_generalist_with_custom_num_classes(self, mock_glob, mock_get_dataloaders, mock_specialize):
        """
        Tests that the specialization phase correctly identifies a generalist
        model and skips specializing it, even when the number of classes is
        not the default of 10. This test is expected to FAIL until the
        hardcoded '10' is fixed.
        """
        # Arrange
        # 0. Prevent loading of incompatible pretrained models from other tests
        mock_glob.return_value = []

        # 1. Mock get_dataloaders to return a custom number of classes (e.g., 5)
        custom_num_classes = 5
        mock_get_dataloaders.return_value = (None, None, None, custom_num_classes)

        # 2. Write a standard config file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        # 3. Initialize the simulator. This will set self.num_classes to 5
        # and will call `specialize` on its newly created population.
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.num_classes, custom_num_classes)

        # 3a. Reset the mock to ignore the calls from the initialization phase.
        mock_specialize.reset_mock()

        # 4. Create a "generalist" model wrapper for this custom configuration.
        # Its niche covers all classes from 0 to 4.
        from src.model_wrapper import ModelWrapper
        from src.enums import ModelName
        generalist_wrapper = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=list(range(custom_num_classes)),
            device='cpu',
            num_classes=custom_num_classes
        )
        simulator.population = [generalist_wrapper]

        # Act
        # Run the specialization phase for a generation > 0
        simulator._run_specialization_phase(generation=1)

        # Assert
        # The `specialize` function should NOT have been called, because the model
        # is a generalist. This will fail if the check is hardcoded to `range(10)`.
        mock_specialize.assert_not_called()


    def test_sequential_constructive_merge_with_zero_validation_split_raises_error(self):
        """
        Tests that the simulator raises a ValueError at initialization if the
        'sequential_constructive' merge strategy is selected with a
        validation_split of 0.
        """
        # Arrange
        self.base_config['merge_strategy'] = 'sequential_constructive'
        self.base_config['validation_split'] = 0.0
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        # Act & Assert
        with self.assertRaises(ValueError) as cm:
            EvolutionSimulator(config_path=self.config_path)

        self.assertIn(
            "The 'sequential_constructive' merge strategy requires a validation_split > 0",
            str(cm.exception),
            "The simulator did not raise the expected ValueError for an invalid configuration."
        )


if __name__ == '__main__':
    unittest.main()