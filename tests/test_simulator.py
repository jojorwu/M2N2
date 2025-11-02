import unittest
import os
import shutil
import yaml
import sys
import torch
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model_wrapper import ModelWrapper
from src.model_factory import create_model
from src.constants import COMMAND_FILE, FITNESS_LOG_FILENAME

class TestSimulatorInitialization(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_simulator_test"
        os.makedirs(self.test_dir, exist_ok=True)
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        self.base_config = {
            'model_name': 'CIFAR10', 'dataset_name': 'CIFAR10', 'precision_config': '32',
            'num_generations': 1, 'population_size': 2, 'num_offspring': 2, 'mate_selection_strategy': 'healing',
            'generation_strategy': 'replace_worst', 'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'optimizer_config': {'learning_rate': 0.001}, 'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 4, 'mutation_rate': 0.1, 'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 1.0, 'subset_percentage': 0.01,
            'validation_split': 0.1, 'default_epochs': {'specialize': 0, 'finetune': 0},
            'delete_old_models': True, 'seed': 12345
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        self.model_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if os.path.exists(COMMAND_FILE):
            os.remove(COMMAND_FILE)
        if os.path.exists(FITNESS_LOG_FILENAME):
            os.remove(FITNESS_LOG_FILENAME)
        if os.path.exists('fitness_history.png'):
            os.remove('fitness_history.png')

    @patch('src.simulator.glob.glob')
    @patch('src.simulator.ModelWrapper.from_file')
    def test_simulator_loads_population(self, mock_from_file, mock_glob):
        mock_glob.return_value = ['model1.pth', 'model2.pth']
        mock_wrapper = MagicMock(spec=ModelWrapper)
        mock_from_file.return_value = mock_wrapper

        simulator = EvolutionSimulator(config_path=self.config_path)

        self.assertEqual(len(simulator.population), 2)
        self.assertEqual(mock_from_file.call_count, 2)

    def test_save_final_population_preserves_survivors_and_removes_unrelated(self):
        user_file_path = os.path.join(self.model_dir, "user_backup.pth")
        torch.save({}, user_file_path)

        loaded_model_path = os.path.join(self.model_dir, "loaded_and_replaced.pth")
        torch.save({}, loaded_model_path)

        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.loaded_model_files = [loaded_model_path]

        final_model = ModelWrapper(
            model=create_model(self.base_config['model_name'], 10, 'cpu'),
            device='cpu', model_name=self.base_config['model_name'], niche_classes=[0]
        )
        final_model.fitness = 99.88
        simulator.population = [final_model]

        simulator._save_final_population(model_dir=self.model_dir)

        final_model_filename = "model_niche_0_fitness_99.88.pth"
        final_model_path = os.path.join(self.model_dir, final_model_filename)

        self.assertTrue(os.path.exists(final_model_path), "Final model was not saved.")
        self.assertTrue(os.path.exists(user_file_path), "User file was deleted.")
        self.assertFalse(os.path.exists(loaded_model_path), "The original loaded model file was not deleted.")

    def test_simulator_restarts_and_clears_artifacts(self):
        with open(FITNESS_LOG_FILENAME, "w") as f: f.write("dummy_log")

        simulator = EvolutionSimulator(config_path=self.config_path)

        with open(COMMAND_FILE, 'w') as f:
            json.dump({'restart_simulation': True}, f)

        simulator._handle_commands()

        # After a restart, the log should be re-initialized with only the header
        self.assertTrue(os.path.exists(FITNESS_LOG_FILENAME))
        with open(FITNESS_LOG_FILENAME, 'r') as f:
            content = f.read().strip()
        self.assertEqual(content, "generation,best_fitness,average_fitness")

    def test_simulator_loads_dynamic_config_from_command_file(self):
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.config_manager.merge_strategy, 'average')

        command_data = {'merge_strategy': 'fitness_weighted'}
        with open(COMMAND_FILE, 'w') as f:
            json.dump(command_data, f)

        simulator._handle_commands()

        self.assertEqual(simulator.config_manager.merge_strategy, 'fitness_weighted', "Merge strategy was not dynamically updated.")
        from src.merge_strategies import FitnessWeightedMergeStrategy
        self.assertIsInstance(simulator.merge_strategy, FitnessWeightedMergeStrategy, "Simulator's merge strategy object was not re-initialized.")

if __name__ == '__main__':
    unittest.main()
