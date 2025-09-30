import unittest
import os
import shutil
import yaml
import sys
from torchvision import transforms

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2n2_implementation.simulator import EvolutionSimulator

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

    def test_resnet_config_initializes_and_applies_resize_transform(self):
        """
        Tests that setting model_config to RESNET correctly initializes the
        simulator without crashing and applies the necessary Resize transform
        to the data.
        """
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

if __name__ == '__main__':
    unittest.main()