import unittest
import os
import shutil
import yaml
import sys
import torch
from unittest.mock import patch
import builtins

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model_wrapper import ModelWrapper
from src.model import CifarCNN
from src.constants import FITNESS_LOG_FILENAME

class TestSimulatorReliability(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_reliability_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        self.base_config = {
            'model_name': 'CNN', 'dataset_name': 'CIFAR10',
            'precision_config': '32', 'num_generations': 1, 'population_size': 2, 'num_offspring': 2,
            'mate_selection_strategy': 'healing', 'generation_strategy': 'replace_worst',
            'merge_strategy': 'average', 'fitness_weighted_merge_dampening_factor': 25.0,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 4, 'mutation_rate': 0.1, 'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 1.0, 'subset_percentage': 0.01,
            'validation_split': 0.1, 'default_epochs': {'specialize': 1, 'finetune': 1},
            'delete_old_models': True,
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        # Suppress the logger setup for all tests in this class to prevent
        # interference with unittest's `assertLogs`.
        self.setup_logger_patcher = patch('src.simulator.setup_logger')
        self.mock_setup_logger = self.setup_logger_patcher.start()

    def tearDown(self):
        self.setup_logger_patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(FITNESS_LOG_FILENAME):
            os.remove(FITNESS_LOG_FILENAME)

    def test_evaluation_phase_handles_empty_population_gracefully(self):
        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.population = []
        with self.assertLogs('M2N2_SIMULATOR', level='ERROR') as cm:
            simulator._run_evaluation_phase()
        self.assertTrue(any("Population is empty. Cannot evaluate." in msg for msg in cm.output))

    def test_delete_old_models_preserves_survivors_and_removes_replaced(self):
        model_dir = os.path.join(self.test_dir, "models")
        os.makedirs(model_dir)

        surviving_model_path = os.path.join(model_dir, "model_niche_1_fitness_95.00.pth")
        torch.save(CifarCNN().state_dict(), surviving_model_path)

        replaced_model_path = os.path.join(model_dir, "model_to_be_deleted.pth")
        torch.save(CifarCNN().state_dict(), replaced_model_path)

        intermediate_model_path = os.path.join(model_dir, "model_niche_2_fitness_50.00.pth")
        torch.save(CifarCNN().state_dict(), intermediate_model_path)

        survivor_wrapper = ModelWrapper(
            model_name=self.base_config['model_name'],
            model=CifarCNN(),
            niche_classes=[1],
            device="cpu"
        )
        survivor_wrapper.fitness = 95.00

        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.population = [survivor_wrapper]
        # Crucially, the loaded files list must contain the full paths
        simulator.loaded_model_files = [surviving_model_path, replaced_model_path]

        simulator._delete_old_models(model_dir)

        self.assertTrue(os.path.exists(surviving_model_path),
                        "The surviving loaded model file should be preserved.")
        self.assertTrue(os.path.exists(replaced_model_path),
                        "A non-surviving loaded model file should also be preserved.")
        self.assertFalse(os.path.exists(intermediate_model_path),
                         "An intermediate, non-loaded model file should be deleted.")

    def test_fitness_logging_handles_os_error_gracefully(self):
        original_open = builtins.open

        def selective_mock(file, *args, **kwargs):
            if file == FITNESS_LOG_FILENAME:
                raise OSError("Disk full!")
            return original_open(file, *args, **kwargs)

        with patch('builtins.open', side_effect=selective_mock):
            with self.assertLogs('M2N2_SIMULATOR', level='WARNING') as cm:
                EvolutionSimulator(config_path=self.config_path)
        self.assertTrue(any("Could not create fitness log" in msg for msg in cm.output))

        simulator = EvolutionSimulator(config_path=self.config_path)
        with patch('builtins.open', side_effect=selective_mock):
            with self.assertLogs('M2N2_SIMULATOR', level='WARNING') as cm:
                simulator._log_fitness_to_csv(gen=1, best=10.0, avg=5.0)
        self.assertTrue(any("Failed to write to fitness log" in msg for msg in cm.output))

    def test_evolution_phase_handles_empty_population_gracefully(self):
        """
        Ensures that the evolution phase does not crash if the population is
        empty, and logs an appropriate error.
        """
        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.population = []  # Manually empty the population

        with self.assertLogs('M2N2_SIMULATOR', level='ERROR') as cm:
            simulator._run_evolution_phase()
        self.assertTrue(any("Population is empty. Skipping evolution." in msg for msg in cm.output))

    def test_save_final_population_handles_os_error_gracefully(self):
        """
        Ensures _save_final_population logs a warning and continues if a
        model save fails with an OSError.
        """
        simulator = EvolutionSimulator(config_path=self.config_path)

        # Create a mock model wrapper that will fail to save
        mock_wrapper = unittest.mock.MagicMock(spec=ModelWrapper)
        mock_wrapper.save.side_effect = OSError("Disk is full")
        # Configure attributes needed for filename generation
        mock_wrapper.niche_classes = [0]
        mock_wrapper.fitness = 50.0

        simulator.population = [mock_wrapper]

        with self.assertLogs('M2N2_SIMULATOR', level='WARNING') as cm:
            simulator._save_final_population(model_dir=self.test_dir)
            # Verify the warning was logged
            self.assertTrue(any("Failed to save model" in msg for msg in cm.output))
            # Verify the mock save method was called
            mock_wrapper.save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
