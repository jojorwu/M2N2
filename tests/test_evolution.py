import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
import os
import random
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evolution import ModelWrapper, merge, select_mates, create_next_generation
from src.model import CifarCNN
from src.selection_strategies import HealingMateSelectionStrategy
from src.merge_strategies import (
    FitnessWeightedMergeStrategy,
    LayerWiseMergeStrategy,
    SequentialConstructiveMergeStrategy
)
from src.generation_strategies import ReplaceWorstStrategy
from src.utils import set_seed
import src.selection_strategies

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
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 85.0
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        parent2.fitness = 15.0
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)
            for param in parent2.model.parameters():
                param.fill_(0.0)
        child = merge(parent1, parent2, strategy=FitnessWeightedMergeStrategy())
        dampening_factor = 25.0
        dampened_fitness1 = parent1.fitness + dampening_factor
        dampened_fitness2 = parent2.fitness + dampening_factor
        total_dampened_fitness = dampened_fitness1 + dampened_fitness2
        expected_weight1 = dampened_fitness1 / total_dampened_fitness
        expected_child_tensor_val = expected_weight1
        child_param = next(child.model.parameters())
        self.assertTrue(
            torch.allclose(child_param, torch.full_like(child_param, expected_child_tensor_val)),
            f"Child weights are incorrect. Expected ~{expected_child_tensor_val:.4f}, but got {child_param.mean():.4f}."
        )

    @patch('src.model_wrapper.ModelWrapper._calculate_accuracy')
    @patch('src.model.models.resnet18')
    def test_sequential_constructive_merge_skips_parameterless_resnet_layers(self, mock_resnet_constructor, mock_calculate_accuracy):
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
        mock_calculate_accuracy.return_value = 50.0
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)
        merge(parent1, parent2, strategy=SequentialConstructiveMergeStrategy(), validation_loader=dummy_loader)
        expected_calls = 4
        self.assertEqual(
            mock_calculate_accuracy.call_count,
            expected_calls,
            f"The validation function was called {mock_calculate_accuracy.call_count} times, but {expected_calls} were expected."
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
        child1 = merge(parent1, parent2, strategy=LayerWiseMergeStrategy(seed=seed1))
        child2 = merge(parent1, parent2, strategy=LayerWiseMergeStrategy(seed=seed1))
        child3 = merge(parent1, parent2, strategy=LayerWiseMergeStrategy(seed=seed2))
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

    @patch('src.model_wrapper.ModelWrapper.evaluate_by_class')
    def test_healing_selection_handles_multiple_weakest_classes(self, mock_evaluate_by_class):
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
        strategy = HealingMateSelectionStrategy()
        for _ in range(30):
            _, parent2 = strategy.select_mates(population, dataset_name='CIFAR10')
            selected_weakest_classes.append(parent2.niche_classes[0])
        unique_selected = set(selected_weakest_classes)
        self.assertTrue(len(unique_selected) > 1, "Mate selection appears biased.")
        self.assertEqual(unique_selected, expected_weakest_indices, "The selected weakest classes do not match the expected set.")

    @patch('src.model_wrapper.ModelWrapper._calculate_accuracy')
    def test_sequential_constructive_merge_handles_variable_num_classes(self, mock_calculate_accuracy):
        num_classes = 5
        mock_calculate_accuracy.return_value = 50.0
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device, num_classes=num_classes)
        parent1.fitness = 80.0
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device, num_classes=num_classes)
        parent2.fitness = 70.0
        child = merge(parent1, parent2, strategy=SequentialConstructiveMergeStrategy(), validation_loader=dummy_loader)
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

    @patch('src.model_wrapper.get_dataloaders')
    def test_evaluate_uses_subset_percentage(self, mock_get_dataloaders):
        mock_get_dataloaders.return_value = (None, None, "dummy_test_loader", 10)
        model_wrapper = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        model_wrapper.fitness_is_current = False
        test_subset_percentage = 0.5
        with patch('src.model_wrapper.ModelWrapper._calculate_accuracy', return_value=50.0):
            model_wrapper.evaluate('CIFAR10', subset_percentage=test_subset_percentage)
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
        next_gen = create_next_generation(population, duplicate_child, population_size, 'CIFAR10', strategy=ReplaceWorstStrategy())
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

    @patch('src.model_wrapper.ModelWrapper.evaluate_by_class')
    def test_healing_selection_fallback_chooses_next_best_distinct_instance(self, mock_evaluate_by_class):
        mock_evaluate_by_class.return_value = [90, 80, 70, 60, 50, 10, 85, 95, 88, 75]
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 95.0
        expected_parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        expected_parent2.fitness = 90.0
        other_model = ModelWrapper(model_name='CIFAR10', niche_classes=[2], device=self.device)
        other_model.fitness = 85.0
        population = [parent1, expected_parent2, other_model]
        strategy = HealingMateSelectionStrategy()
        _, selected_parent2 = strategy.select_mates(population, dataset_name='CIFAR10')
        self.assertIsNot(selected_parent2, parent1, "Parent 2 should not be the same instance as Parent 1.")
        self.assertIs(selected_parent2, expected_parent2, "The fallback did not select the next-best distinct model instance.")

    def test_layer_wise_merge_on_resnet_is_not_all_or_nothing(self):
        seed = 42
        parent1 = ModelWrapper(model_name='RESNET', niche_classes=[0], device=self.device)
        parent2 = ModelWrapper(model_name='RESNET', niche_classes=[1], device=self.device)
        with torch.no_grad():
            for param in parent1.model.parameters():
                param.fill_(1.0)
            for param in parent2.model.parameters():
                param.fill_(0.0)
        child = merge(parent1, parent2, strategy=LayerWiseMergeStrategy(seed=seed))
        child_sd = child.model.state_dict()
        parent1_sd = parent1.model.state_dict()
        parent2_sd = parent2.model.state_dict()
        self.assertFalse(are_state_dicts_equal(child_sd, parent1_sd), "Child's weights are identical to Parent 1. No layers were mixed.")
        self.assertFalse(are_state_dicts_equal(child_sd, parent2_sd), "Child's weights are identical to Parent 2. No layers were mixed.")

    def test_generate_and_verify_sequential_constructive_merge(self):
        import torch
        import os
        seed = 123
        golden_file_path = 'tests/golden_sequential_merge.pth'
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        parent1.fitness = 90.0
        parent2.fitness = 80.0
        with torch.no_grad():
            for i, param in enumerate(parent1.model.parameters()):
                param.fill_(float(i + 1))
            for param in parent2.model.parameters():
                param.fill_(0.0)
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)
        mock_fitness_sequence = [50.0, 55.0, 45.0, 60.0, 58.0, 65.0]
        with patch('src.model_wrapper.ModelWrapper._calculate_accuracy', side_effect=mock_fitness_sequence):
            set_seed(seed)
            child = merge(parent1, parent2, strategy=SequentialConstructiveMergeStrategy(), validation_loader=dummy_loader)
        if not os.path.exists(golden_file_path):
            torch.save(child.model.state_dict(), golden_file_path)
            self.skipTest("Golden reference file created. Re-run tests to verify against it.")
        else:
            golden_state_dict = torch.load(golden_file_path)
            self.assertTrue(
                are_state_dicts_equal(child.model.state_dict(), golden_state_dict),
                "The output of the optimized strategy does not match the golden reference."
            )
            os.remove(golden_file_path)

    @patch('src.evolution.tqdm')
    @patch('src.evolution.get_dataloaders')
    @patch('src.evolution.optim.Adam')
    def test_specialize_handles_progress_bar_toggle(self, mock_adam, mock_get_dataloaders, mock_tqdm):
        dummy_batch = (torch.randn(1, 3, 32, 32), torch.randint(0, 10, (1,)))
        mock_get_dataloaders.return_value = ([dummy_batch], None, None, 10)
        mock_tqdm.return_value.__iter__.return_value = iter([dummy_batch])
        model_wrapper = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        from src.evolution import specialize
        specialize(model_wrapper, dataset_name='CIFAR10', epochs=1, show_progress_bar=True)
        mock_tqdm.assert_called_once()
        self.assertTrue(mock_tqdm.return_value.set_postfix.called)
        mock_tqdm.reset_mock()
        specialize(model_wrapper, dataset_name='CIFAR10', epochs=1, show_progress_bar=False)
        mock_tqdm.assert_not_called()

    @patch('src.model_wrapper.ModelWrapper.evaluate_by_class')
    def test_healing_selection_fallback_skips_identical_clone(self, mock_evaluate_by_class):
        mock_evaluate_by_class.return_value = [90, 80, 70, 60, 50, 10, 85, 95, 88, 75]
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 95.0
        clone_of_parent1 = copy.deepcopy(parent1)
        clone_of_parent1.fitness = 95.0
        expected_parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        expected_parent2.fitness = 90.0
        population = [parent1, clone_of_parent1, expected_parent2]
        strategy = HealingMateSelectionStrategy()
        selected_parent1, selected_parent2 = strategy.select_mates(population, dataset_name='CIFAR10')
        self.assertNotEqual(selected_parent1, selected_parent2, "Selected parents should be genetically different.")
        self.assertEqual(selected_parent2, expected_parent2, "The fallback did not select the next-best genetically distinct model.")

    @patch('src.model_wrapper.ModelWrapper.evaluate_by_class')
    def test_healing_selection_fallback_handles_fitness_ties(self, mock_evaluate_by_class):
        mock_evaluate_by_class.return_value = [10] * 10
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 90.0
        clone = copy.deepcopy(parent1)
        clone.fitness = 90.0
        distinct_model = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        distinct_model.fitness = 85.0
        population = [parent1, clone, distinct_model]
        random.shuffle(population)
        strategy = HealingMateSelectionStrategy()
        _, selected_parent2 = strategy.select_mates(population, dataset_name='CIFAR10')
        self.assertIsNot(selected_parent2, parent1, "Parent 2 should not be the same instance as Parent 1.")
        self.assertNotEqual(selected_parent2, parent1, "Parent 2 should not be a deep copy of Parent 1.")
        self.assertEqual(selected_parent2, distinct_model, "The fallback did not select the correct distinct model.")

    def test_sequential_constructive_merge_uses_single_batch_optimization(self):
        parent1 = ModelWrapper(model_name='CIFAR10', niche_classes=[0], device=self.device)
        parent1.fitness = 90.0
        parent2 = ModelWrapper(model_name='CIFAR10', niche_classes=[1], device=self.device)
        parent2.fitness = 80.0
        dummy_batch = (torch.randn(1, 3, 32, 32), torch.randint(0, 10, (1,)))
        dummy_loader = torch.utils.data.DataLoader([dummy_batch, "dummy_batch_2"], batch_size=1)
        with patch('src.model_wrapper.ModelWrapper._calculate_accuracy', return_value=50.0) as mock_calculate_accuracy:
            merge(parent1, parent2, strategy=SequentialConstructiveMergeStrategy(), validation_loader=dummy_loader)
        self.assertGreater(mock_calculate_accuracy.call_count, 1, "Validation was not performed for sequential merge.")
        for call in mock_calculate_accuracy.call_args_list:
            _, kwargs = call
            self.assertIn('batch', kwargs, "The 'batch' argument was not provided to _calculate_accuracy.")
            self.assertIsNotNone(kwargs['batch'], "The provided 'batch' argument was None.")

    @patch('src.evolution.torch.cuda.amp.autocast')
    @patch('torch.Tensor.to')
    def test_autocast_is_correctly_enabled_for_mixed_precision(self, mock_tensor_to, mock_autocast):
        from src.evolution import _run_training_epoch
        # Mocking necessary components
        model_wrapper_cuda = MagicMock()
        model_wrapper_cuda.device = 'cuda'
        model_wrapper_cuda.model.return_value = torch.randn(1, 10)

        model_wrapper_cpu = MagicMock()
        model_wrapper_cpu.device = 'cpu'
        model_wrapper_cpu.model.return_value = torch.randn(1, 10)

        mock_optimizer = MagicMock()
        data_tensor = torch.randn(1, 3, 32, 32)
        target_tensor = torch.randint(0, 10, (1,))
        dummy_loader = [(data_tensor, target_tensor)]
        mock_scaler = MagicMock()

        # Define the side effect for the mock_tensor_to
        mock_tensor_to.side_effect = [data_tensor, target_tensor] * 3

        # Case 1: 16-bit precision on CUDA
        _run_training_epoch(model_wrapper_cuda, mock_optimizer, dummy_loader, mock_scaler, '16', 'test')
        mock_autocast.assert_called_with(enabled=True)

        # Case 2: 32-bit precision on CUDA
        _run_training_epoch(model_wrapper_cuda, mock_optimizer, dummy_loader, mock_scaler, '32', 'test')
        mock_autocast.assert_called_with(enabled=False)

        # Case 3: 16-bit precision on CPU
        _run_training_epoch(model_wrapper_cpu, mock_optimizer, dummy_loader, mock_scaler, '16', 'test')
        mock_autocast.assert_called_with(enabled=False)

if __name__ == '__main__':
    unittest.main()
