import unittest
from unittest.mock import patch
import os
import shutil
import yaml
import sys
import pytest
from torchvision import transforms

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator

@pytest.mark.slow
class TestResnetIntegration(unittest.TestCase):
    """
    Integration test to ensure the RESNET configuration works end-to-end,
    including correct data loader creation and model instantiation.
    """

    def setUp(self):
        """Set up a temporary environment for the simulator test."""
        self.test_dir = "tests/temp_integration_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a config file that uses the RESNET model with the CIFAR10 dataset
        self.config = {
            'model_config': 'RESNET',
            'dataset_name': 'CIFAR10',
            'model_dir': os.path.join(self.test_dir, 'models'),
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 1, # Keep population small for speed
            'merge_strategy': 'average',
            'mutation_rate': 0.0,
            'initial_mutation_strength': 0.0,
            'mutation_decay_factor': 1.0,
            'subset_percentage': 0.01, # Use a tiny subset
            'validation_split': 0.5,
            'default_epochs': {'specialize': 0, 'finetune': 0},
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        """Clean up the temporary environment after the test."""
        shutil.rmtree(self.test_dir)

    @patch('src.simulator.glob.glob')
    def test_resnet_config_initializes_and_applies_resize_transform(self, mock_glob):
        """
        Tests that setting model_config to RESNET correctly initializes the
        simulator without crashing and applies the necessary Resize transform
        to the data.
        """
        # Arrange
        # Prevent loading incompatible models from other tests
        mock_glob.return_value = []

        # Act
        try:
            # This will crash if the dataset_name is not correctly decoupled
            # from the model_config.
            simulator = EvolutionSimulator(config_path=self.config_path)
        except ValueError as e:
            self.fail(f"EvolutionSimulator failed to initialize with RESNET config: {e}")

        # Assert
        # 1. Check that the simulator was created
        self.assertIsInstance(simulator, EvolutionSimulator)

        # 2. Inspect the transforms on the validation loader's dataset
        # We need to traverse the nested Subset objects to get to the original dataset
        current_dataset = simulator.validation_loader.dataset
        while hasattr(current_dataset, 'dataset'):
            current_dataset = current_dataset.dataset

        validation_transforms = current_dataset.transform.transforms

        # 3. Check if the Resize transform is present
        has_resize_transform = any(isinstance(t, transforms.Resize) for t in validation_transforms)
        self.assertTrue(
            has_resize_transform,
            "The Resize transform was not added to the dataloader for the RESNET model."
        )

        # 4. Check if the Resize transform has the correct size
        resize_transform = next(t for t in validation_transforms if isinstance(t, transforms.Resize))
        self.assertEqual(
            resize_transform.size,
            (224, 224),
            f"Resize transform has incorrect size. Expected (224, 224), got {resize_transform.size}."
        )

    def test_simulator_runs_for_multiple_generations(self):
        """
        Tests that the simulator can run for more than one generation without
        crashing due to missing arguments in the `create_next_generation` call.
        """
        # Arrange
        # Modify the config to run for 2 generations and use a simple model
        self.config['num_generations'] = 2
        self.config['population_size'] = 2 # At least 2 for mate selection
        self.config['model_config'] = 'CIFAR10'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

        # Act
        try:
            # This would crash at the end of generation 1 with the bug
            simulator = EvolutionSimulator(config_path=self.config_path)
            simulator.run()
        except TypeError as e:
            self.fail(f"Simulator crashed with TypeError, likely due to the bug in create_next_generation: {e}")
        except Exception as e:
            self.fail(f"Simulator crashed with an unexpected exception: {e}")

        # Assert
        # The simulator should have recorded fitness for 2 generations
        self.assertEqual(
            len(simulator.fitness_history),
            2,
            "Simulator did not complete 2 generations as expected."
        )


    def test_imagenet_config_creates_correct_model_and_data(self):
        """
        Tests that using the 'IMAGENET' dataset config correctly initializes
        a ResNet model with 1000 classes and a data loader that produces
        224x224 images.
        """
        # Arrange
        # 1. Modify the config to use the new IMAGENET dataset
        self.config['dataset_name'] = 'IMAGENET'
        self.config['model_config'] = 'RESNET' # ResNet is required for this image size
        self.config['dataset_configs'] = {
            'IMAGENET': {'num_classes': 1000}
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

        # Act
        # 2. Initialize the simulator with this config
        simulator = EvolutionSimulator(config_path=self.config_path, seed=1337)

        # Assert
        # 3. Check the model's output classes
        self.assertEqual(
            simulator.population[0].model.num_classes,
            1000,
            "Model was not initialized with 1000 classes for IMAGENET config."
        )

        # 4. Check the data loader's image dimensions
        # Get one batch from the validation loader
        data_batch, _ = next(iter(simulator.validation_loader))
        # The shape should be (N, C, H, W) -> (batch_size, 3, 224, 224)
        self.assertEqual(
            data_batch.shape[2],
            224,
            f"Image height is not 224, got {data_batch.shape[2]} instead."
        )
        self.assertEqual(
            data_batch.shape[3],
            224,
            f"Image width is not 224, got {data_batch.shape[3]} instead."
        )


if __name__ == '__main__':
    unittest.main()