import unittest
from unittest.mock import patch
import torch
import sys
import os
from torchvision import datasets, transforms

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import get_dataloaders
from src.enums import DatasetName

class TestDataFiltering(unittest.TestCase):
    """Unit tests for data loading and filtering logic."""

    @classmethod
    def setUpClass(cls):
        """Pre-load the dataset to find indices for the test."""
        # This is done once for the class to speed up tests.
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            # Find 50 indices that are guaranteed NOT to be class '3' (cat).
            cls.non_cat_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label != 3][:50]
        except Exception as e:
            # If dataset download fails in a CI environment, skip these tests.
            raise unittest.SkipTest(f"Skipping data tests due to dataset loading error: {e}")


    @patch('numpy.random.permutation')
    def test_niche_filtering_is_applied_before_subset_sampling(self, mock_permutation):
        """
        Tests that niche filtering is applied BEFORE subset sampling.
        This test deterministically proves the bug by controlling the "random"
        subset to ensure it contains no niche samples. The test should fail
        with the original implementation and pass with the fix.
        """
        # Arrange
        # Force the random permutation to return our pre-selected non-cat indices.
        mock_permutation.return_value = self.non_cat_indices

        # Act
        # Request the 'cat' niche, with a subset percentage that would normally
        # select 50 samples.
        train_loader, _, _, _ = get_dataloaders(
            dataset_name=DatasetName.CIFAR10,
            niche_classes=[3],       # The 'cat' class
            subset_percentage=0.001, # This will select 50 samples due to the mock
            validation_split=0,
            seed=42                  # Seed is irrelevant due to the mock
        )

        # Assert
        # Correct behavior: First filter for all 5000 cats, then take a 0.1% subset (~5 cats). The loader is NOT empty.
        # Buggy behavior: First take a 0.1% subset (our 50 non-cat images), then filter for cats. The loader IS empty.
        # Therefore, this assertion will fail if the logic is buggy.
        self.assertGreater(
            len(train_loader.dataset),
            0,
            "The train loader is empty. This confirms that subset sampling was incorrectly applied before niche filtering."
        )

if __name__ == '__main__':
    unittest.main()