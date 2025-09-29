import pytest
import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2n2_implementation.evolution import ModelWrapper, merge

# Define a fixture to create two parent models for testing
@pytest.fixture
def create_parents():
    """A pytest fixture to create two parent ModelWrapper instances."""
    # Use 'CIFAR10' as the representative model type for testing merge logic
    parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device='cpu')
    parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device='cpu')

    # Assign some fitness values for fitness-based strategies
    parent1.fitness = 80.0
    parent2.fitness = 60.0

    return parent1, parent2

# Parametrize the test to run for each merge strategy
@pytest.mark.parametrize("strategy", [
    'average',
    'fitness_weighted',
    'layer-wise',
    'sequential_constructive'
])
def test_merge_strategies(create_parents, strategy):
    """
    Tests that each merge strategy runs without error and produces a valid child.
    """
    parent1, parent2 = create_parents

    # The sequential_constructive strategy requires a fitness score to be calculated
    # for the parents first. We can mock this by setting the fitness attribute,
    # which is already done in the fixture. For other strategies, it's not strictly
    # necessary but doesn't hurt.

    # Execute the merge function
    if strategy == 'sequential_constructive':
        # This strategy requires a validation loader, so we create a dummy one.
        from m2n2_implementation.data import get_dataloaders
        _, validation_loader, _ = get_dataloaders(dataset_name='CIFAR10', batch_size=2, subset_percentage=0.01)
        child_wrapper = merge(parent1, parent2, strategy=strategy, validation_loader=validation_loader)
    else:
        child_wrapper = merge(parent1, parent2, strategy=strategy)

    # --- Assertions ---

    # 1. Check if a valid ModelWrapper object was returned
    assert isinstance(child_wrapper, ModelWrapper), f"Merge strategy '{strategy}' did not return a ModelWrapper object."

    # 2. Check if the child's model has the same architecture (i.e., same state_dict keys)
    parent_keys = parent1.model.state_dict().keys()
    child_keys = child_wrapper.model.state_dict().keys()
    assert parent_keys == child_keys, f"Child model from '{strategy}' has different architecture than parents."

    # 3. Check that the child's weights are not the same as either parent's weights
    # (This is a sanity check to ensure some form of merging happened)
    parent1_sd = parent1.model.state_dict()
    child_sd = child_wrapper.model.state_dict()

    # Note: For sequential_constructive, it's possible the child is identical to the
    # fitter parent if no layer swaps were beneficial. This is expected behavior.
    if strategy != 'sequential_constructive':
        # Check at least one parameter is different
        are_different = any(not torch.equal(child_sd[key], parent1_sd[key]) for key in child_keys)
        assert are_different, f"Child's weights are identical to parent1's for '{strategy}' strategy."

def test_merge_fitness_weighted_with_zero_fitness():
    """
    Tests the 'fitness_weighted' strategy specifically for the case where both
    parents have zero fitness, ensuring it falls back to a simple average.
    """
    parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device='cpu')
    parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device='cpu')
    parent1.fitness = 0.0
    parent2.fitness = 0.0

    child = merge(parent1, parent2, strategy='fitness_weighted')

    # Check if the child's weights are the exact average of the parents'
    parent1_sd = parent1.model.state_dict()
    parent2_sd = parent2.model.state_dict()
    child_sd = child.model.state_dict()

    for key in child_sd:
        expected_tensor = (parent1_sd[key] + parent2_sd[key]) / 2.0
        assert torch.allclose(child_sd[key], expected_tensor), \
            f"Weight mismatch for key {key} in zero-fitness weighted merge."