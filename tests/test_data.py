import unittest
from unittest.mock import patch
import torch
import sys
import os
from torchvision import datasets, transforms

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import get_dataloaders
from src.enums import DatasetName, ModelName

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
            model_name=ModelName.CIFAR10,
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

    def test_subset_sampling_is_reproducible_with_seed(self):
        """
        Tests that subset sampling is reproducible when a seed is provided.
        """
        # Arrange
        subset_percentage = 0.1
        seed = 42

        # Act
        train_loader1, _, _, _ = get_dataloaders(
            dataset_name=DatasetName.CIFAR10,
            model_name=ModelName.CIFAR10,
            subset_percentage=subset_percentage,
            validation_split=0,
            seed=seed
        )

        # In a real-world scenario, other code might run here and affect the
        # global numpy random state. We simulate this by calling another
        # function that uses numpy.random.
        import numpy
        numpy.random.rand(10)

        train_loader2, _, _, _ = get_dataloaders(
            dataset_name=DatasetName.CIFAR10,
            model_name=ModelName.CIFAR10,
            subset_percentage=subset_percentage,
            validation_split=0,
            seed=seed
        )

        # Assert
        indices1 = train_loader1.dataset.indices
        indices2 = train_loader2.dataset.indices
        self.assertEqual(list(indices1), list(indices2),
                         "The data subsets are not identical, indicating a reproducibility issue.")

    def test_dataloader_shuffling_is_reproducible_with_fix(self):
        """
        Tests that the DataLoader shuffling is reproducible after fixing the
        shared generator bug. This test would fail with the original code.
        """
        # Arrange
        seed = 888
        # Act
        train_loader1, _, _, _ = get_dataloaders(
            dataset_name=DatasetName.CIFAR10, model_name=ModelName.CIFAR10,
            validation_split=0.2, seed=seed, subset_percentage=0.1, batch_size=32
        )
        train_loader2, _, _, _ = get_dataloaders(
            dataset_name=DatasetName.CIFAR10, model_name=ModelName.CIFAR10,
            validation_split=0.2, seed=seed, subset_percentage=0.1, batch_size=32
        )
        # Assert
        order1 = [torch.mean(batch[0]).item() for batch in train_loader1]
        order2 = [torch.mean(batch[0]).item() for batch in train_loader2]
        self.assertEqual(order1, order2,
                         "DataLoader shuffling is not reproducible, the fix was not successful.")

    def test_get_dataloaders_does_not_affect_global_random_state(self):
        """
        Tests that get_dataloaders does not affect the global random state,
        ensuring it is a pure function with respect to global seeding.
        """
        # Arrange
        seed = 111
        torch.manual_seed(seed)
        # Get the initial state of the global generator
        initial_state = torch.get_rng_state()

        # Act
        # Call the function, which should use its own local, seeded generators
        # and leave the global generator untouched.
        get_dataloaders(
            dataset_name=DatasetName.CIFAR10, model_name=ModelName.CIFAR10,
            validation_split=0.2, seed=seed, subset_percentage=0.1
        )

        # Assert
        # Get the state of the global generator after the function call
        final_state = torch.get_rng_state()
        # The states should be identical, proving the global state was not used.
        self.assertTrue(torch.equal(initial_state, final_state),
                        "The function altered the global torch random state.")

    def test_warning_for_empty_validation_set(self):
        """
        Tests that a warning is logged when a non-zero validation split
        results in an empty validation set.
        """
        # Arrange
        # Use a very small subset and a small validation split to guarantee
        # that the number of validation samples rounds down to zero.
        # 10 samples * 0.05 = 0.5, which rounds down to 0.
        with self.assertLogs('M2N2_DATALOADER', level='WARNING') as cm:
            # Act
            get_dataloaders(
                dataset_name=DatasetName.CIFAR10,
                model_name=ModelName.CIFAR10,
                subset_percentage=0.0002,  # Approx. 10 samples
                validation_split=0.05,
                seed=42
            )
            # Assert
            self.assertEqual(len(cm.output), 1)
            self.assertIn("Validation set is empty", cm.output[0])

if __name__ == '__main__':
    unittest.main()