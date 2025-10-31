import unittest
import os
import shutil
import torch
import yaml
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model import CifarCNN

class TestModelCleanup(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_cleanup_test"
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
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_delete_old_models_preserves_survivors_and_user_files(self):
        model_dir = os.path.join(self.test_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        user_file_path = os.path.join(model_dir, "user_backup.pth")
        with open(user_file_path, "w") as f: f.write("user data")

        old_model_path = os.path.join(model_dir, "model_niche_0_fitness_5.00.pth")
        torch.save(CifarCNN().state_dict(), old_model_path)

        surviving_model_wrapper = MagicMock()
        surviving_model_wrapper.niche_classes = [1]
        surviving_model_wrapper.fitness = 95.00
        surviving_model_path = os.path.join(model_dir, "model_niche_1_fitness_95.00.pth")
        torch.save(CifarCNN().state_dict(), surviving_model_path)

        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.population = [surviving_model_wrapper]
        simulator.loaded_model_files = [surviving_model_path] # Simulate it was loaded

        simulator._delete_old_models(model_dir)

        self.assertTrue(os.path.exists(user_file_path), "User file was deleted.")
        self.assertTrue(os.path.exists(surviving_model_path), "Surviving model was deleted.")
        self.assertFalse(os.path.exists(old_model_path), "Old model was not deleted.")

if __name__ == '__main__':
    unittest.main()
