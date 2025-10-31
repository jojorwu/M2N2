import unittest
import os
import shutil
import yaml
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evolution import specialize
from src.model_wrapper import ModelWrapper
from src.config_manager import ConfigManager
from src.model import CifarCNN
from src.enums import ModelName

class TestEmptyLoader(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_empty_loader_test"
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
            'batch_size': 4,
            'mutation_rate': 0.1,
            'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 1.0,
            'subset_percentage': 0.01,
            'validation_split': 0.1,
            'default_epochs': {'specialize': 1, 'finetune': 1},
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        self.config_manager = ConfigManager(config_path=self.config_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('src.evolution._run_training_session')
    @patch('src.evolution.get_dataloaders')
    def test_training_is_skipped_for_empty_loader(self, mock_get_dataloaders, mock_run_training_session):
        """
        Tests that the training session is skipped gracefully if the dataloader is empty.
        """
        # Arrange: Mock get_dataloaders to return an empty list as the train_loader
        mock_get_dataloaders.return_value = ([], None, None, 10)

        model = CifarCNN(num_classes=10)
        model_wrapper = ModelWrapper(
            model_name=ModelName.CIFAR10,
            model=model,
            niche_classes=[0, 1],
            device='cpu'
        )

        # Act & Assert
        with self.assertLogs('M2N2_SIMULATOR', level='WARNING') as cm:
            specialize(model_wrapper, self.config_manager)
            self.assertTrue(any("Skipping specialize" in msg for msg in cm.output))
            self.assertTrue(any("data loader is empty" in msg for msg in cm.output))

        # Assert that the core training session function was never called
        mock_run_training_session.assert_not_called()

if __name__ == '__main__':
    unittest.main()
