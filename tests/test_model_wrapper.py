import unittest
import sys
import os
import torch

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_wrapper import ModelWrapper
from src.enums import ModelName

class TestModelWrapper(unittest.TestCase):
    """Unit tests for the ModelWrapper class."""

    def test_equality_and_hash_are_order_independent(self):
        """
        Tests that two ModelWrapper instances with the same niche classes
        but in a different order are considered equal and have the same hash.
        This verifies that the niche comparison is order-independent.
        """
        # Arrange
        # Create two ModelWrapper instances that are identical except for the
        # order of their niche classes.
        wrapper1 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=[1, 0],
            device='cpu'
        )
        wrapper2 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=[0, 1],
            device='cpu'
        )

        # To ensure the models themselves are identical, we'll copy the state
        # from the first model to the second.
        wrapper2.model.load_state_dict(wrapper1.model.state_dict())

        # Act & Assert
        # 1. Test for equality: The wrappers should be considered equal.
        self.assertEqual(wrapper1, wrapper2, "ModelWrappers with the same niche classes in a different order should be equal.")

        # 2. Test for hash consistency: Their hashes should also be equal.
        self.assertEqual(hash(wrapper1), hash(wrapper2), "Hashes of ModelWrappers with the same niche classes in a different order should be equal.")

    def test_inequality_for_different_niches(self):
        """
        Tests that two ModelWrapper instances with different niches are
        not considered equal.
        """
        # Arrange
        wrapper1 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=[0],
            device='cpu'
        )
        wrapper2 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=[1],
            device='cpu'
        )

        # Assert
        self.assertNotEqual(wrapper1, wrapper2, "ModelWrappers with different niches should not be equal.")

    def test_inequality_for_different_model_states(self):
        """
        Tests that two ModelWrapper instances with different model weights
        are not considered equal.
        """
        # Arrange
        wrapper1 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=[0],
            device='cpu'
        )
        # Create a second wrapper, which will have different initial weights
        wrapper2 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=[0],
            device='cpu'
        )

        # Assert
        self.assertNotEqual(wrapper1, wrapper2, "ModelWrappers with different model states should not be equal.")


if __name__ == '__main__':
    unittest.main()