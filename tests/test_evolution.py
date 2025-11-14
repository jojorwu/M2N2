import unittest
from unittest.mock import patch
import torch
import sys
import os
import random

# Add the project root to the Python path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evolution import ModelWrapper, merge, select_mates
from src.model import CifarCNN

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

    @patch('src.merge_strategies._get_validation_fitness')
    def test_random_half_merge_swaps_layers_and_chooses_fitter(self, mock_get_validation_fitness):
        """
        Tests that the 'RandomHalfMergeStrategy' (formerly sequential_constructive)
        swaps roughly half the layers and correctly chooses the better model.
        """
        # Arrange
        # 1. Create two parents with easily trackable weights (all 1s and all 0s)
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 80.0
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)

        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        parent2.fitness = 70.0
        with torch.no_grad():
            for param in parent2.model.parameters():
                param.fill_(0.0)

        # 2. A dummy validation loader is required
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)

        # 3. Configure the mock validation to simulate the hybrid being better
        # The first call is for the base model (parent1), the second for the hybrid.
        mock_get_validation_fitness.side_effect = [50.0, 60.0]

        # Act
        child = merge(parent1, parent2, strategy='sequential_constructive', validation_loader=dummy_loader)

        # Assert
        # 1. Check that the fitness check was called twice (base and hybrid)
        self.assertEqual(mock_get_validation_fitness.call_count, 2)

        # 2. Check that some, but not all, layers were swapped.
        # The child's weights should be a mix of 0s and 1s.
        child_params = list(child.model.parameters())
        has_zeros = any(torch.any(p == 0.0) for p in child_params)
        has_ones = any(torch.any(p == 1.0) for p in child_params)
        self.assertTrue(has_zeros and has_ones, "The child model does not appear to be a hybrid of the parents.")

        # 3. Reset mock and test the case where the fitter parent is better
        mock_get_validation_fitness.reset_mock()
        mock_get_validation_fitness.side_effect = [60.0, 50.0]
        child = merge(parent1, parent2, strategy='sequential_constructive', validation_loader=dummy_loader)
        child_params = list(child.model.parameters())
        is_all_ones = all(torch.all(p == 1.0) for p in child_params)
        self.assertTrue(is_all_ones, "The merge should have returned the fitter parent, but it appears to have returned a hybrid.")

    def test_layer_wise_merge_is_deterministic_with_seed(self):
        """
        Tests that the 'layer-wise' merge strategy produces identical models
        when given the same seed, and different models with different seeds.
        """
        # Arrange
        seed1 = 42
        seed2 = 1337

        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)

        # Assign easily trackable weights
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)
            for param in parent2.model.parameters():
                param.fill_(0.0)

        # Act
        # Merge twice with the same seed
        child1 = merge(parent1, parent2, strategy='layer-wise', seed=seed1)
        child2 = merge(parent1, parent2, strategy='layer-wise', seed=seed1)

        # Merge once with a different seed
        child3 = merge(parent1, parent2, strategy='layer-wise', seed=seed2)

        # Assert
        # 1. The two children created with the same seed should be identical
        child1_params = list(child1.model.parameters())
        child2_params = list(child2.model.parameters())
        self.assertEqual(len(child1_params), len(child2_params))
        for p1, p2 in zip(child1_params, child2_params):
            self.assertTrue(torch.equal(p1, p2), "Models created with the same seed are not identical.")

        # 2. The child created with a different seed should be different
        child3_params = list(child3.model.parameters())
        is_different = False
        for p1, p3 in zip(child1_params, child3_params):
            if not torch.equal(p1, p3):
                is_different = True
                break
        self.assertTrue(is_different, "Model created with a different seed was not different.")


    def test_select_mates_handles_multiple_weakest_classes(self):
        """
        Tests that if the best model has multiple classes with the same lowest
        accuracy, the mate selection process will randomly choose from among
        them, rather than deterministically picking the first one.
        """
        # Arrange
        accuracies = [90, 80, 70, 50, 60, 85, 50, 95, 88, 75]
        expected_weakest_indices = {3, 6}

        population = []
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[], device=self.device)
        parent1.fitness = 90.0
        parent1.per_class_fitness = accuracies # Set the cached value
        population.append(parent1)

        for i in range(10):
            specialist = ModelWrapper(model_name='CIFAR10', niche_classes=[i], device=self.device)
            specialist.fitness = 20.0
            population.append(specialist)

        # Act
        random.seed(42)
        selected_weakest_classes = []
        for _ in range(30):
            _, parent2 = select_mates(population)
            selected_weakest_classes.append(parent2.niche_classes[0])

        # Assert
        unique_selected = set(selected_weakest_classes)
        self.assertTrue(len(unique_selected) > 1, "Mate selection appears biased.")
        self.assertEqual(unique_selected, expected_weakest_indices, "The selected weakest classes do not match the expected set.")


    @patch('src.merge_strategies._get_validation_fitness')
    def test_sequential_constructive_merge_handles_variable_num_classes(self, mock_get_validation_fitness):
        """
        Tests that the 'sequential_constructive' merge strategy correctly
        infers the number of classes from the parent models instead of
        using a hardcoded value of 10.
        """
        # Arrange
        num_classes = 5 # Use a non-10 number of classes
        mock_get_validation_fitness.return_value = 50.0
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)

        # Create parents with a custom number of classes
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device, num_classes=num_classes)
        parent1.fitness = 80.0

        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device, num_classes=num_classes)
        parent2.fitness = 70.0

        # Act
        child = merge(parent1, parent2, strategy='sequential_constructive', validation_loader=dummy_loader)

        # Assert
        # The child's niche classes should be a list from 0 to num_classes-1
        self.assertEqual(
            child.niche_classes,
            list(range(num_classes)),
            f"Child's niche classes should be a range up to {num_classes}, but got {child.niche_classes}."
        )


    @patch('src.data.get_dataloaders')
    @patch('src.evolution._calculate_loss')
    @patch('torch.optim.lr_scheduler.ReduceLROnPlateau')
    def test_finetune_uses_reduce_lr_on_plateau(self, mock_scheduler_class, mock_calculate_loss, mock_get_dataloaders):
        """
        Tests that the `finetune` function correctly uses the ReduceLROnPlateau
        scheduler by stepping it with the calculated validation loss.
        """
        # Arrange
        # 1. Mock the behavior of the dependencies
        mock_scheduler_instance = mock_scheduler_class.return_value
        mock_calculate_loss.return_value = 0.123  # A dummy validation loss
        mock_get_dataloaders.return_value = (None, None, None, 10) # Prevent actual data loading

        # 2. Create the necessary inputs for the finetune function
        model_wrapper = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        dummy_validation_loader = "dummy_loader" # Can be a simple string as it's just passed through

        # Act
        # 3. Call the finetune function
        from src.evolution import finetune
        finetune(model_wrapper, 'CIFAR10', dummy_validation_loader, epochs=1)

        # Assert
        # 4. Verify that the scheduler was created and its step method was called
        self.assertTrue(mock_scheduler_class.called, "ReduceLROnPlateau scheduler was not created.")
        self.assertTrue(mock_scheduler_instance.step.called, "Scheduler's step() method was not called.")

        # 5. Verify that the step method was called with the correct validation loss
        mock_scheduler_instance.step.assert_called_once_with(0.123)


    @patch('src.evolution.get_dataloaders')
    def test_evaluate_uses_subset_percentage(self, mock_get_dataloaders):
        """
        Tests that the `evaluate` function correctly passes the
        `subset_percentage` argument to the `get_dataloaders` call.
        """
        # Arrange
        mock_get_dataloaders.return_value = (None, None, "dummy_test_loader", 10)
        model_wrapper = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        model_wrapper.fitness_is_current = False # Ensure evaluation is not skipped
        test_subset_percentage = 0.5

        # Act
        from src.evolution import evaluate
        with patch('src.evolution._calculate_accuracy', return_value=50.0):
             evaluate(model_wrapper, 'CIFAR10', subset_percentage=test_subset_percentage)


        # Assert
        self.assertTrue(mock_get_dataloaders.called, "get_dataloaders was not called.")
        call_args, call_kwargs = mock_get_dataloaders.call_args
        self.assertEqual(
            call_kwargs.get('subset_percentage'),
            test_subset_percentage,
            f"get_dataloaders was called with subset_percentage={call_kwargs.get('subset_percentage')}, "
            f"but {test_subset_percentage} was expected."
        )

    def test_create_next_generation_avoids_duplicates(self):
        """
        Tests that `create_next_generation` does not add a child if it's an
        exact duplicate of a model already in the population.
        """
        # Arrange
        from src.evolution import create_next_generation
        import copy

        population_size = 5
        population = [ModelWrapper(model_name='CIFAR10', niche_classes=[i], device=self.device) for i in range(population_size)]
        for i, p in enumerate(population):
            p.fitness = 70.0 - i * 10 # Assign descending fitness

        # Create a child that is a perfect duplicate of the second-best model
        duplicate_child = copy.deepcopy(population[1])
        duplicate_child.fitness = population[1].fitness # Ensure fitness is also identical
        # Mark fitness as current to prevent evaluate() from running and changing the fitness
        duplicate_child.fitness_is_current = True


        # Act
        next_gen = create_next_generation(population, duplicate_child, population_size, 'CIFAR10')

        # Assert
        # The population size should not have grown
        self.assertEqual(len(next_gen), population_size)

        # Count how many times the duplicate appears in the next generation
        duplicate_count = sum(1 for model in next_gen if model == duplicate_child)
        self.assertEqual(duplicate_count, 1, "A duplicate model was added to the new generation.")


if __name__ == '__main__':
    unittest.main()