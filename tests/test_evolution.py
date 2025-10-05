import unittest
from unittest.mock import patch
import torch
import sys
import os
import random
import copy

# Add the project root to the Python path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evolution import ModelWrapper, merge, select_mates
from src.model import CifarCNN

def are_state_dicts_equal(dict1, dict2):
    """A helper function to compare two model state dictionaries."""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True

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
        """
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 85.0
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        parent2.fitness = 15.0
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)
            for param in parent2.model.parameters():
                param.fill_(0.0)
        child = merge(parent1, parent2, strategy='fitness_weighted')
        dampening_factor = 25.0
        dampened_fitness1 = parent1.fitness + dampening_factor
        dampened_fitness2 = parent2.fitness + dampening_factor
        total_dampened_fitness = dampened_fitness1 + dampened_fitness2
        expected_weight1 = dampened_fitness1 / total_dampened_fitness
        expected_child_tensor_val = expected_weight1
        child_param = next(child.model.parameters())
        self.assertTrue(
            torch.allclose(child_param, torch.full_like(child_param, expected_child_tensor_val)),
            f"Child weights are incorrect. Expected ~{expected_child_tensor_val:.4f}, but got {child_param.mean():.4f}. "
            "The specialist parent's contribution is likely being diluted."
        )

    @patch('src.merge_strategies._get_validation_fitness')
    @patch('src.model.models.resnet18')
    def test_sequential_constructive_merge_skips_parameterless_resnet_layers(self, mock_resnet_constructor, mock_get_validation_fitness):
        class MockResNetModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()
                self.layer1 = torch.nn.Linear(10, 10)
                self.maxpool = torch.nn.MaxPool2d(2)
                self.fc = torch.nn.Linear(10, 10)
            def forward(self, x): return x
        mock_resnet_constructor.return_value = MockResNetModule()
        parent1 = ModelWrapper(model_name='RESNET', niche_classes=[0], device=self.device)
        parent1.fitness = 80.0
        parent2 = ModelWrapper(model_name='RESNET', niche_classes=[1], device=self.device)
        parent2.fitness = 20.0
        mock_get_validation_fitness.return_value = 50.0
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)
        merge(parent1, parent2, strategy='sequential_constructive', validation_loader=dummy_loader)
        expected_calls = 4
        self.assertEqual(
            mock_get_validation_fitness.call_count,
            expected_calls,
            f"The validation function was called {mock_get_validation_fitness.call_count} times, but {expected_calls} were expected. "
            "It may not be correctly skipping parameter-less layers."
        )

    def test_layer_wise_merge_is_deterministic_with_seed(self):
        seed1 = 42
        seed2 = 1337
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)
            for param in parent2.model.parameters():
                param.fill_(0.0)
        child1 = merge(parent1, parent2, strategy='layer-wise', seed=seed1)
        child2 = merge(parent1, parent2, strategy='layer-wise', seed=seed1)
        child3 = merge(parent1, parent2, strategy='layer-wise', seed=seed2)
        child1_params = list(child1.model.parameters())
        child2_params = list(child2.model.parameters())
        self.assertEqual(len(child1_params), len(child2_params))
        for p1, p2 in zip(child1_params, child2_params):
            self.assertTrue(torch.equal(p1, p2), "Models created with the same seed are not identical.")
        child3_params = list(child3.model.parameters())
        is_different = False
        for p1, p3 in zip(child1_params, child3_params):
            if not torch.equal(p1, p3):
                is_different = True
                break
        self.assertTrue(is_different, "Model created with a different seed was not different.")

    @patch('src.evolution.evaluate_by_class')
    def test_select_mates_handles_multiple_weakest_classes(self, mock_evaluate_by_class):
        accuracies = [90, 80, 70, 50, 60, 85, 50, 95, 88, 75]
        mock_evaluate_by_class.return_value = accuracies
        expected_weakest_indices = {3, 6}
        population = []
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[], device=self.device)
        parent1.fitness = 90.0
        population.append(parent1)
        for i in range(10):
            specialist = ModelWrapper(model_name='CIFAR10', niche_classes=[i], device=self.device)
            specialist.fitness = 20.0
            population.append(specialist)
        random.seed(42)
        selected_weakest_classes = []
        for _ in range(30):
            _, parent2 = select_mates(population, dataset_name='CIFAR10')
            selected_weakest_classes.append(parent2.niche_classes[0])
        unique_selected = set(selected_weakest_classes)
        self.assertTrue(len(unique_selected) > 1, "Mate selection appears biased.")
        self.assertEqual(unique_selected, expected_weakest_indices, "The selected weakest classes do not match the expected set.")

    @patch('src.merge_strategies._get_validation_fitness')
    def test_sequential_constructive_merge_handles_variable_num_classes(self, mock_get_validation_fitness):
        num_classes = 5
        mock_get_validation_fitness.return_value = 50.0
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device, num_classes=num_classes)
        parent1.fitness = 80.0
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device, num_classes=num_classes)
        parent2.fitness = 70.0
        child = merge(parent1, parent2, strategy='sequential_constructive', validation_loader=dummy_loader)
        self.assertEqual(child.niche_classes, list(range(num_classes)), f"Child's niche classes should be a range up to {num_classes}, but got {child.niche_classes}.")

    @patch('src.data.get_dataloaders')
    @patch('src.evolution._calculate_loss')
    @patch('torch.optim.lr_scheduler.ReduceLROnPlateau')
    def test_finetune_uses_reduce_lr_on_plateau(self, mock_scheduler_class, mock_calculate_loss, mock_get_dataloaders):
        mock_scheduler_instance = mock_scheduler_class.return_value
        mock_calculate_loss.return_value = 0.123
        mock_get_dataloaders.return_value = (None, None, None, 10)
        model_wrapper = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        dummy_validation_loader = "dummy_loader"
        from src.evolution import finetune
        finetune(model_wrapper, 'CIFAR10', dummy_validation_loader, epochs=1)
        self.assertTrue(mock_scheduler_class.called, "ReduceLROnPlateau scheduler was not created.")
        self.assertTrue(mock_scheduler_instance.step.called, "Scheduler's step() method was not called.")
        mock_scheduler_instance.step.assert_called_once_with(0.123)

    @patch('src.evolution.get_dataloaders')
    def test_evaluate_uses_subset_percentage(self, mock_get_dataloaders):
        mock_get_dataloaders.return_value = (None, None, "dummy_test_loader", 10)
        model_wrapper = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        model_wrapper.fitness_is_current = False
        test_subset_percentage = 0.5
        from src.evolution import evaluate
        with patch('src.evolution._calculate_accuracy', return_value=50.0):
             evaluate(model_wrapper, 'CIFAR10', subset_percentage=test_subset_percentage)
        self.assertTrue(mock_get_dataloaders.called, "get_dataloaders was not called.")
        call_args, call_kwargs = mock_get_dataloaders.call_args
        self.assertEqual(call_kwargs.get('subset_percentage'), test_subset_percentage, f"get_dataloaders was called with subset_percentage={call_kwargs.get('subset_percentage')}, but {test_subset_percentage} was expected.")

    def test_create_next_generation_avoids_duplicates(self):
        from src.evolution import create_next_generation
        population_size = 5
        population = [ModelWrapper(model_name='CIFAR10', niche_classes=[i], device=self.device) for i in range(population_size)]
        for i, p in enumerate(population):
            p.fitness = 70.0 - i * 10
        duplicate_child = copy.deepcopy(population[1])
        duplicate_child.fitness = population[1].fitness
        duplicate_child.fitness_is_current = True
        next_gen = create_next_generation(population, duplicate_child, population_size, 'CIFAR10')
        self.assertEqual(len(next_gen), population_size)
        duplicate_count = sum(1 for model in next_gen if model == duplicate_child)
        self.assertEqual(duplicate_count, 1, "A duplicate model was added to the new generation.")

    def test_model_wrapper_hashing(self):
        wrapper1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        wrapper2 = copy.deepcopy(wrapper1)
        wrapper3 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        self.assertEqual(wrapper1, wrapper2, "Deepcopied wrappers should be equal.")
        self.assertEqual(hash(wrapper1), hash(wrapper2), "Hashes of equal wrappers should be equal.")
        self.assertNotEqual(wrapper1, wrapper3, "Wrappers with different niches should not be equal.")
        self.assertNotEqual(hash(wrapper1), hash(wrapper3), "Hashes of unequal wrappers should not be equal.")
        model_set = {wrapper1, wrapper2}
        self.assertEqual(len(model_set), 1, "A set should not contain duplicate ModelWrappers.")
        model_set.add(wrapper3)
        self.assertEqual(len(model_set), 2, "A set should be able to contain different ModelWrappers.")

    @patch('src.evolution.evaluate_by_class')
    def test_select_mates_fallback_chooses_next_best_distinct_instance(self, mock_evaluate_by_class):
        """
        Tests that the fallback mate selection logic correctly selects the
        next-best model by fitness that is not the same instance as Parent 1.
        """
        # --- Arrange ---
        # Mock class evaluation to force the fallback mechanism by making class 5
        # the weakest, but we will not provide a specialist for it.
        mock_evaluate_by_class.return_value = [90, 80, 70, 60, 50, 10, 85, 95, 88, 75]

        # Create Parent 1 (the best model)
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 95.0

        # Create a second model, which will be the expected Parent 2.
        # It has a lower fitness than Parent 1.
        expected_parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        expected_parent2.fitness = 90.0

        # Create a third, lower-fitness model that should not be selected.
        other_model = ModelWrapper(model_name='CIFAR10', niche_classes=[2], device=self.device)
        other_model.fitness = 85.0

        # The population is sorted by fitness: parent1, expected_parent2, other_model
        population = [parent1, expected_parent2, other_model]

        # --- Act ---
        # The logic should:
        # 1. Select parent1 as the best model.
        # 2. Identify class 5 as its weakest.
        # 3. Fail to find a specialist for class 5.
        # 4. Fall back to the sorted list.
        # 5. Skip parent1 (as it's the same instance).
        # 6. Select expected_parent2 as it's the next in the list and a different instance.
        _, selected_parent2 = select_mates(population, dataset_name='CIFAR10')

        # --- Assert ---
        self.assertIsNot(selected_parent2, parent1, "Parent 2 should not be the same instance as Parent 1.")
        self.assertIs(selected_parent2, expected_parent2, "The fallback did not select the next-best distinct model instance.")

    def test_layer_wise_merge_on_resnet_is_not_all_or_nothing(self):
        """
        Tests that the 'layer-wise' merge on a ResNet actually swaps
        individual layers, rather than the entire model. This test now
        compares state dictionaries directly to avoid metadata mismatches.
        """
        seed = 42
        parent1 = ModelWrapper(model_name='RESNET', niche_classes=[0], device=self.device)
        parent2 = ModelWrapper(model_name='RESNET', niche_classes=[1], device=self.device)
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)
            for param in parent2.model.parameters():
                param.fill_(0.0)
        child = merge(parent1, parent2, strategy='layer-wise', seed=seed)
        child_sd = child.model.state_dict()
        parent1_sd = parent1.model.state_dict()
        parent2_sd = parent2.model.state_dict()
        self.assertFalse(are_state_dicts_equal(child_sd, parent1_sd), "Child's weights are identical to Parent 1. No layers were mixed.")
        self.assertFalse(are_state_dicts_equal(child_sd, parent2_sd), "Child's weights are identical to Parent 2. No layers were mixed.")

    def test_generate_and_verify_sequential_constructive_merge(self):
        """
        This test serves two purposes:
        1. When run against the original code, it generates a 'golden reference'
           file of the output from the sequential_constructive merge.
        2. When run against the optimized code, it verifies that the output is
           identical to the golden reference.
        """
        # --- Arrange ---
        import torch
        import os

        # Use a fixed seed for reproducibility
        seed = 123
        golden_file_path = 'tests/golden_sequential_merge.pth'

        # Create two distinct parent models
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        parent1.fitness = 90.0
        parent2.fitness = 80.0
        with torch.no_grad():
            for i, param in enumerate(parent1.model.parameters()):
                param.fill_(float(i + 1)) # Parent1 has weights 1.0, 2.0, ...
            for param in parent2.model.parameters():
                param.fill_(0.0)          # Parent2 has weights 0.0

        # Create a dummy validation loader
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)

        # Mock the validation fitness to return a sequence of values that
        # will cause some layers to be swapped and others to be kept.
        # The sequence is: initial, conv1 (keep), conv2 (reject), fc1 (keep), fc2 (reject), fc3 (keep)
        mock_fitness_sequence = [50.0, 55.0, 45.0, 60.0, 58.0, 65.0]

        # --- Act ---
        with patch('src.merge_strategies._get_validation_fitness', side_effect=mock_fitness_sequence):
            child = merge(parent1, parent2, strategy='sequential_constructive', validation_loader=dummy_loader, seed=seed)

        # --- Assert / Verify ---
        if not os.path.exists(golden_file_path):
            print(f"\n[INFO] Golden reference file not found. Creating '{golden_file_path}'...")
            torch.save(child.model.state_dict(), golden_file_path)
            self.skipTest("Golden reference file created. Re-run tests to verify against it.")
        else:
            print(f"\n[INFO] Golden reference file found. Verifying output...")
            golden_state_dict = torch.load(golden_file_path)
            self.assertTrue(
                are_state_dicts_equal(child.model.state_dict(), golden_state_dict),
                "The output of the optimized strategy does not match the golden reference."
            )
            # Clean up the file after a successful test run
            os.remove(golden_file_path)


    @patch('src.evolution.tqdm')
    @patch('src.evolution.get_dataloaders')
    @patch('src.evolution.optim.Adam')
    def test_specialize_handles_progress_bar_toggle(self, mock_adam, mock_get_dataloaders, mock_tqdm):
        """
        Tests that the specialize function correctly shows or hides the
        tqdm progress bar based on the 'show_progress_bar' flag.
        """
        # --- Arrange ---
        # Mock the dataloader to return a single dummy batch to ensure the loop runs
        dummy_batch = (torch.randn(1, 3, 32, 32), torch.randint(0, 10, (1,)))
        mock_get_dataloaders.return_value = ([dummy_batch], None, None, 10)

        # Configure the mock tqdm object to be iterable so the training loop runs
        mock_tqdm.return_value.__iter__.return_value = iter([dummy_batch])

        model_wrapper = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)

        # --- Act & Assert (Case 1: Progress bar enabled) ---
        from src.evolution import specialize
        specialize(
            model_wrapper, dataset_name='CIFAR10', epochs=1, show_progress_bar=True
        )
        mock_tqdm.assert_called_once()
        # Check that the progress bar's postfix was updated
        self.assertTrue(mock_tqdm.return_value.set_postfix.called)

        # --- Act & Assert (Case 2: Progress bar disabled) ---
        mock_tqdm.reset_mock()
        specialize(
            model_wrapper, dataset_name='CIFAR10', epochs=1, show_progress_bar=False
        )
        mock_tqdm.assert_not_called()


if __name__ == '__main__':
    unittest.main()