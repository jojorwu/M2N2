import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import torch
import yaml
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model import CifarCNN
from src.config_manager import ConfigManager

class TestSimulatorInitialization(unittest.TestCase):
    """Unit tests for the EvolutionSimulator's initialization logic."""

    def setUp(self):
        self.test_dir = "tests/temp_simulator_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)
        self.base_config = {
            'model_config': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 1,
            'mate_selection_strategy': 'healing',
            'generation_strategy': 'replace_worst',
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
        shutil.rmtree(self.test_dir)
        if os.path.exists("command_config.json"):
            os.remove("command_config.json")
        model_dir = "src/pretrained_models"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    def test_save_final_population_preserves_unrelated_files(self):
        model_dir = "src/pretrained_models"
        os.makedirs(model_dir, exist_ok=True)
        loaded_model_path = os.path.join(model_dir, "model_niche_0_fitness_10.00.pth")
        user_file_path = os.path.join(model_dir, "user_backup_model.pth")
        dummy_model = CifarCNN()
        torch.save(dummy_model.state_dict(), loaded_model_path)
        with open(user_file_path, "w") as f:
            f.write("This is a user backup, do not delete.")
        config = self.base_config.copy()
        config['num_generations'] = 1
        config['population_size'] = 1
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.run()
        self.assertTrue(os.path.exists(user_file_path), "The user's unrelated file was deleted.")
        self.assertFalse(os.path.exists(loaded_model_path), "The original loaded model file was not deleted.")
        new_model_files = [f for f in os.listdir(model_dir) if f.startswith('model_niche_')]
        self.assertGreater(len(new_model_files), 0, "No new model was saved to the directory.")

    def test_clear_simulation_artifacts_deletes_log_file(self):
        log_file_path = "fitness_log.csv"
        with open(log_file_path, "w") as f:
            f.write("generation,best_fitness,average_fitness\n")
            f.write("1,10.0,5.0\n")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator._clear_simulation_artifacts()
        self.assertFalse(os.path.exists(log_file_path), "The fitness_log.csv file was not deleted.")

    @patch('src.simulator.glob.glob')
    def test_loaded_model_fitness_is_marked_as_stale(self, mock_glob):
        model_dir = os.path.join(self.test_dir, "pretrained_models")
        os.makedirs(model_dir, exist_ok=True)
        dummy_model_path = os.path.join(model_dir, "model_niche_0_fitness_99.9.pth")
        torch.save(CifarCNN().state_dict(), dummy_model_path)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        mock_glob.return_value = [dummy_model_path]
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertGreater(len(simulator.population), 0, "Simulator failed to load any models.")
        self.assertFalse(simulator.population[0].fitness_is_current)

    @patch('src.simulator.setup_logger')
    def test_logging_is_configurable(self, mock_setup_logger):
        config_with_log = self.base_config.copy()
        log_file_path = os.path.join(self.test_dir, "test.log")
        config_with_log['log_file'] = log_file_path
        with open(self.config_path, 'w') as f:
            yaml.dump(config_with_log, f)
        EvolutionSimulator(config_path=self.config_path)
        mock_setup_logger.assert_called_with(log_file=log_file_path)

        config_without_log = self.base_config.copy()
        config_without_log['log_file'] = None
        with open(self.config_path, 'w') as f:
            yaml.dump(config_without_log, f)
        EvolutionSimulator(config_path=self.config_path)
        mock_setup_logger.assert_called_with(log_file=None)

    def test_simulator_uses_seed_from_config(self):
        config_with_seed = self.base_config.copy()
        expected_seed = 12345
        config_with_seed['seed'] = expected_seed
        with open(self.config_path, 'w') as f:
            yaml.dump(config_with_seed, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.config_manager.seed, expected_seed, "Simulator did not use the seed from the config file.")

    @patch('src.config_manager.np.random.randint', return_value=54321)
    def test_simulator_generates_random_seed_if_not_provided(self, mock_randint):
        config_without_seed = self.base_config.copy()
        if 'seed' in config_without_seed:
            del config_without_seed['seed']
        with open(self.config_path, 'w') as f:
            yaml.dump(config_without_seed, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.config_manager.seed, 54321, "Simulator did not generate a random seed when none was provided.")
        mock_randint.assert_called_once()

    @patch('src.simulator.EvolutionSimulator._run_specialization_phase')
    @patch('src.simulator.EvolutionSimulator._run_evaluation_phase')
    @patch('src.simulator.select_mates', return_value=(None, None))
    def test_simulator_loads_dynamic_config_from_command_file(self, mock_select_mates, mock_evaluation, mock_specialization):
        config = self.base_config.copy()
        config['merge_strategy'] = 'average'
        config['mutation_rate'] = 0.1
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        command_config = {"merge_strategy": "fitness_weighted", "mutation_rate": 0.99}
        with open("command_config.json", 'w') as f:
            json.dump(command_config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.config_manager.merge_strategy, 'average')
        self.assertEqual(simulator.config_manager.mutation_rate, 0.1)
        simulator.run()
        self.assertEqual(simulator.config_manager.merge_strategy, 'fitness_weighted', "Merge strategy was not dynamically updated.")
        self.assertEqual(simulator.config_manager.mutation_rate, 0.99, "Mutation rate was not dynamically updated.")
        mock_select_mates.assert_called_once()

    @patch('src.simulator.specialize')
    @patch('src.simulator.get_dataloaders')
    @patch('src.simulator.glob.glob')
    def test_specialization_is_skipped_for_generalist_with_custom_num_classes(self, mock_glob, mock_get_dataloaders, mock_specialize):
        mock_glob.return_value = []
        custom_num_classes = 5
        mock_get_dataloaders.return_value = (None, None, None, custom_num_classes)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.num_classes, custom_num_classes)
        mock_specialize.reset_mock()
        from src.model_wrapper import ModelWrapper
        from src.enums import ModelName
        generalist_wrapper = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=list(range(custom_num_classes)),
            device='cpu',
            num_classes=custom_num_classes
        )
        simulator.population = [generalist_wrapper]
        simulator._run_specialization_phase(generation=1)
        mock_specialize.assert_not_called()

    def test_sequential_constructive_merge_with_zero_validation_split_raises_error(self):
        self.base_config['merge_strategy'] = 'sequential_constructive'
        self.base_config['validation_split'] = 0.0
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        with self.assertRaises(ValueError) as cm:
            EvolutionSimulator(config_path=self.config_path)
        self.assertIn("The 'sequential_constructive' merge strategy requires a validation_split > 0", str(cm.exception))

    @patch('src.simulator.EvolutionSimulator._initialize_population')
    def test_cleanup_deletes_loaded_models_when_flag_is_true(self, mock_initialize_population):
        model_dir = "src/pretrained_models"
        os.makedirs(model_dir, exist_ok=True)
        loaded_model_to_delete = os.path.join(model_dir, "model_niche_0_fitness_10.0.pth")
        user_file_to_preserve = os.path.join(model_dir, "user_backup.pth")
        torch.save(CifarCNN().state_dict(), loaded_model_to_delete)
        with open(user_file_to_preserve, "w") as f:
            f.write("preserve this file")
        config = self.base_config.copy()
        config['delete_old_models'] = True
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.loaded_model_files = [loaded_model_to_delete]
        from src.model_wrapper import ModelWrapper
        simulator.population = [ModelWrapper(model_name=simulator.config_manager.model_config, niche_classes=[0], device=simulator.device, num_classes=simulator.num_classes)]
        simulator.population[0].fitness = 99.0
        simulator._save_final_population()
        self.assertFalse(os.path.exists(loaded_model_to_delete), "Loaded model was not deleted.")
        self.assertTrue(os.path.exists(user_file_to_preserve), "User backup file was incorrectly deleted.")

    @patch('src.simulator.EvolutionSimulator._initialize_population')
    def test_cleanup_preserves_old_models_when_flag_is_false(self, mock_initialize_population):
        model_dir = "src/pretrained_models"
        os.makedirs(model_dir, exist_ok=True)
        stale_model_path = os.path.join(model_dir, "model_niche_stale_fitness_0.00.pth")
        torch.save(CifarCNN().state_dict(), stale_model_path)
        config = self.base_config.copy()
        config['delete_old_models'] = False
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        from src.model_wrapper import ModelWrapper
        simulator.population = [ModelWrapper(model_name=simulator.config_manager.model_config, niche_classes=[0], device=simulator.device, num_classes=simulator.num_classes)]
        simulator.population[0].fitness = 99.0
        simulator._save_final_population()
        self.assertTrue(os.path.exists(stale_model_path), "Stale model was deleted when flag was false.")
        new_model_files = [f for f in os.listdir(model_dir) if f.startswith('model_niche_')]
        self.assertGreater(len(new_model_files), 1, "New model was not saved alongside the old one.")

if __name__ == '__main__':
    unittest.main()
