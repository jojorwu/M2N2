import unittest
import torch
import sys
import os

# Add the project root to the Python path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2n2_implementation.evolution import ModelWrapper, merge
from m2n2_implementation.model import CifarCNN

class TestEvolution(unittest.TestCase):
    """Unit tests for the evolutionary algorithm components."""

    def setUp(self):
        """Set up common resources for tests."""
        self.device = torch.device("cpu")

    def test_merge_fitness_weighted_with_dampening(self):
        """
        Tests that the 'fitness_weighted' merge strategy uses a dampened
        weighting to prevent a high-fitness parent from completely
        overwhelming a low-fitness (but still valuable) specialist parent.

        This test is designed to FAIL with the original implementation and
        PASS with the corrected, dampened logic.
        """
        # 1. Create two ModelWrappers with a significant fitness disparity
        # This simulates the "healing" scenario where the best model is paired
        # with a specialist in its weakest area.
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 85.0  # High-fitness generalist

        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        parent2.fitness = 15.0  # Low-fitness specialist

        # 2. Assign easily trackable weights to each parent model
        # The fit parent's weights are all 1.0.
        # The specialist parent's weights are all 0.0.
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)
            for param in parent2.model.parameters():
                param.fill_(0.0)

        # 3. Call the merge function with the 'fitness_weighted' strategy
        child = merge(parent1, parent2, strategy='fitness_weighted')

        # 4. Assert the outcome based on the desired "dampened" logic
        #
        # Original (flawed) logic:
        # weight1 = 85 / (85 + 15) = 0.85
        # child_weight = 1.0 * 0.85 + 0.0 * 0.15 = 0.85
        # The specialist's contribution is almost erased.
        #
        # Desired (dampened) logic:
        # A dampening factor (e.g., 25) is added to each fitness score.
        dampening_factor = 25.0
        dampened_fitness1 = parent1.fitness + dampening_factor
        dampened_fitness2 = parent2.fitness + dampening_factor
        total_dampened_fitness = dampened_fitness1 + dampened_fitness2

        expected_weight1 = dampened_fitness1 / total_dampened_fitness # (85+25)/(100+50) = 110/150 = ~0.733

        # Expected child weight = (parent1_weight * expected_weight1) + (parent2_weight * expected_weight2)
        #                       = (1.0 * expected_weight1) + (0.0 * ...) = expected_weight1
        expected_child_tensor_val = expected_weight1

        # Retrieve the first parameter tensor from the child model for verification
        child_param = next(child.model.parameters())

        # Check if the child's weights match the dampened calculation.
        # This assertion will fail until the bug is fixed in evolution.py.
        self.assertTrue(
            torch.allclose(child_param, torch.full_like(child_param, expected_child_tensor_val)),
            f"Child weights are incorrect. Expected ~{expected_child_tensor_val:.4f}, but got {child_param.mean():.4f}. "
            "The specialist parent's contribution is likely being diluted."
        )

if __name__ == '__main__':
    unittest.main()