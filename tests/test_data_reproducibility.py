import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import get_dataloaders
from src.enums import DatasetName, ModelName

class TestDataReproducibility(unittest.TestCase):

    def test_get_dataloaders_is_reproducible(self):
        """
        Tests that calling get_dataloaders twice with the same seed produces
        identical dataloaders, proving the random operations are isolated.
        """
        seed = 42

        # Get a loader with a subset of the dataset
        train_loader_1, val_loader_1, _, _ = get_dataloaders(
            DatasetName.CIFAR10, ModelName.CIFAR10,
            batch_size=64,
            subset_percentage=0.5,
            validation_split=0.1,
            seed=seed
        )

        # Get a second loader with the exact same parameters
        train_loader_2, val_loader_2, _, _ = get_dataloaders(
            DatasetName.CIFAR10, ModelName.CIFAR10,
            batch_size=64,
            subset_percentage=0.5,
            validation_split=0.1,
            seed=seed
        )

        # The first batches should be identical
        batch_1 = next(iter(train_loader_1))
        batch_2 = next(iter(train_loader_2))
        self.assertTrue(torch.equal(batch_1[0], batch_2[0]))

        # The first validation batches should also be identical
        val_batch_1 = next(iter(val_loader_1))
        val_batch_2 = next(iter(val_loader_2))
        self.assertTrue(torch.equal(val_batch_1[0], val_batch_2[0]))

if __name__ == '__main__':
    unittest.main()
