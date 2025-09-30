import unittest
import numpy as np
import sys
import os

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import get_dataloaders

def get_all_indices_from_loader(dataloader):
    """Helper function to extract the original indices from a DataLoader's Subset."""
    # This traverses the nested Subset objects to get to the original indices
    current_dataset = dataloader.dataset
    while hasattr(current_dataset, 'dataset'):
        if hasattr(current_dataset, 'indices'):
             # This is a Subset, so we need to go deeper
            current_dataset = current_dataset.dataset
        else:
            # We've reached the end of the nesting
            break

    if not hasattr(dataloader.dataset, 'indices'):
        return list(range(len(dataloader.dataset)))

    return sorted(dataloader.dataset.indices)


class TestDataDeterminism(unittest.TestCase):
    """Tests the determinism of the data loading and splitting logic."""

    def test_validation_split_is_deterministic_with_seed(self):
        """
        Tests that consecutive calls to get_dataloaders produce the exact
        same train/validation split when provided with the same seed.

        This test is designed to PASS with the fixed implementation.
        """
        # Arrange
        seed = 42 # An arbitrary seed for the test

        # Act
        # Get the first validation loader and its indices using the seed
        _, validation_loader_1, _ = get_dataloaders(
            dataset_name='CIFAR10',
            subset_percentage=0.1,
            validation_split=0.5,
            seed=seed
        )
        indices_1 = get_all_indices_from_loader(validation_loader_1)

        # Get the second validation loader and its indices using the same seed
        _, validation_loader_2, _ = get_dataloaders(
            dataset_name='CIFAR10',
            subset_percentage=0.1,
            validation_split=0.5,
            seed=seed
        )
        indices_2 = get_all_indices_from_loader(validation_loader_2)

        # Assert
        # This should now pass because the seed ensures the shuffle is identical.
        self.assertListEqual(
            indices_1,
            indices_2,
            "Validation set indices are different even when the same seed is provided. "
            "The deterministic splitting logic is not working correctly."
        )

if __name__ == '__main__':
    unittest.main()