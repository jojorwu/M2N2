import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
import os
import random
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evolution import ModelWrapper, merge, select_mates, create_next_generation
from src.model import CifarCNN, ResNetClassifier
from src.enums import ModelName
from src.model_factory import create_model
from src.selection_strategies import HealingMateSelectionStrategy
from src.merge_strategies import (
    FitnessWeightedMergeStrategy,
    LayerWiseMergeStrategy,
    SequentialConstructiveMergeStrategy
)
from src.generation_strategies import ReplaceWorstStrategy
from src.utils import set_seed
from src.config_manager import ConfigManager
import src.selection_strategies

def are_state_dicts_equal(dict1, dict2):
    """A helper function to compare two model state dictionaries."""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True

def create_mock_wrapper(model_name, niche_classes, device, num_classes=10, fitness=0.0):
    """Helper to create a mock ModelWrapper."""
    model = create_model(model_name, num_classes, device)
    wrapper = ModelWrapper(model_name, model, niche_classes, device)
    wrapper.fitness = fitness
    return wrapper

class TestEvolution(unittest.TestCase):
    """Unit tests for the evolutionary algorithm components."""

    def setUp(self):
        """Set up common resources for tests."""
        self.device = torch.device("cpu")

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
        parent1 = create_mock_wrapper(ModelName.RESNET, [0], self.device, fitness=80.0)
        parent2 = create_mock_wrapper(ModelName.RESNET, [1], self.device, fitness=20.0)
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
        parent1 = create_mock_wrapper(ModelName.CIFAR10, [0], self.device)
        parent2 = create_mock_wrapper(ModelName.CIFAR10, [1], self.device)
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

    @patch('src.model_wrapper.ModelWrapper._calculate_accuracy')
    def test_sequential_constructive_merge_handles_variable_num_classes(self, mock_calculate_accuracy):
        num_classes = 5
        mock_calculate_accuracy.return_value = 50.0
        dummy_loader = torch.utils.data.DataLoader([torch.randn(10)], batch_size=1)
        parent1 = create_mock_wrapper(ModelName.CIFAR10, [0], self.device, num_classes=num_classes, fitness=80.0)
        parent2 = create_mock_wrapper(ModelName.CIFAR10, [1], self.device, num_classes=num_classes, fitness=70.0)
        child = merge(parent1, parent2, strategy=SequentialConstructiveMergeStrategy(), validation_loader=dummy_loader)
        self.assertEqual(child.niche_classes, list(range(num_classes)), f"Child's niche classes should be a range up to {num_classes}, but got {child.niche_classes}.")

    @patch('src.data.get_dataloaders')
    @patch('src.evolution._calculate_loss')
    @patch('torch.optim.lr_scheduler.ReduceLROnPlateau')
    def test_finetune_uses_reduce_lr_on_plateau(self, mock_scheduler_class, mock_calculate_loss, mock_get_dataloaders):
        mock_scheduler_instance = mock_scheduler_class.return_value
        mock_calculate_loss.return_value = 0.123
        mock_get_dataloaders.return_value = (None, None, None, 10)
        model_wrapper = create_mock_wrapper(ModelName.CIFAR10, [0], self.device)
        dummy_validation_loader = "dummy_loader"
        from src.evolution import finetune
        from src.config_manager import ConfigManager
        config_manager = ConfigManager('config.yaml')
        config_manager.finetune_epochs = 1
        finetune(model_wrapper, dummy_validation_loader, config_manager)
        self.assertTrue(mock_scheduler_class.called, "ReduceLROnPlateau scheduler was not created.")
        self.assertTrue(mock_scheduler_instance.step.called, "Scheduler's step() method was not called.")
        mock_scheduler_instance.step.assert_called_once_with(0.123)

    @patch('src.model_wrapper.get_dataloaders')
    def test_evaluate_uses_subset_percentage(self, mock_get_dataloaders):
        mock_get_dataloaders.return_value = (None, None, "dummy_test_loader", 10)
        model_wrapper = create_mock_wrapper(ModelName.CIFAR10, [0], self.device)
        model_wrapper.fitness_is_current = False
        test_subset_percentage = 0.5
        with patch('src.model_wrapper.ModelWrapper._calculate_accuracy', return_value=50.0):
            model_wrapper.evaluate('CIFAR10', subset_percentage=test_subset_percentage)
        self.assertTrue(mock_get_dataloaders.called, "get_dataloaders was not called.")
        call_args, call_kwargs = mock_get_dataloaders.call_args
        self.assertEqual(call_kwargs.get('subset_percentage'), test_subset_percentage, f"get_dataloaders was called with subset_percentage={call_kwargs.get('subset_percentage')}, but {test_subset_percentage} was expected.")

    def test_create_next_generation_with_multiple_offspring(self):
        """
        Tests that the elitist strategy correctly selects the best models from
        the current population and a new pool of offspring.
        """
        from src.evolution import create_next_generation
        population_size = 5
        # Population fitness: [90, 80, 70, 60, 50]
        population = [create_mock_wrapper(ModelName.CIFAR10, [i], self.device, fitness=(90.0 - i * 10)) for i in range(population_size)]

        # Offspring pool: one great, one good, one terrible, and one duplicate
        offspring_pool = [
            create_mock_wrapper(ModelName.CIFAR10, [10], self.device, fitness=95.0), # Should be included
            create_mock_wrapper(ModelName.CIFAR10, [11], self.device, fitness=85.0), # Should be included
            create_mock_wrapper(ModelName.CIFAR10, [12], self.device, fitness=45.0), # Should be excluded
            copy.deepcopy(population[1]) # Duplicate of 80.0 fitness model
        ]
        offspring_pool[3].fitness = 80.0
        offspring_pool[3].fitness_is_current = True

        config_manager = MagicMock(spec=ConfigManager)
        config_manager.population_size = population_size
        config_manager.dataset_name = 'CIFAR10'
        config_manager.seed = 42
        config_manager.subset_percentage = 1.0

        # Patch the evaluate method to do nothing, preserving the mock fitness values
        with patch('src.model_wrapper.ModelWrapper.evaluate', return_value=None):
            # Run the generation strategy
            next_gen = create_next_generation(population, offspring_pool, strategy=ReplaceWorstStrategy(), config_manager=config_manager)

        # Verify the results
        self.assertEqual(len(next_gen), population_size, "The next generation has the wrong size.")

        next_gen_fitness = sorted([model.fitness for model in next_gen], reverse=True)
        # Expected fitness: 95 (new), 90 (old), 85 (new), 80 (old), 70 (old)
        expected_fitness = [95.0, 90.0, 85.0, 80.0, 70.0]
        self.assertListEqual(next_gen_fitness, expected_fitness, "The next generation was not composed of the fittest individuals.")

        # Check that the duplicate was handled correctly
        duplicate_count = sum(1 for model in next_gen if model.fitness == 80.0)
        self.assertEqual(duplicate_count, 1, "A duplicate model was incorrectly added to the new generation.")

    def test_model_wrapper_hashing(self):
        wrapper1 = create_mock_wrapper(ModelName.CIFAR10, [0], self.device)
        wrapper2 = copy.deepcopy(wrapper1)
        wrapper3 = create_mock_wrapper(ModelName.CIFAR10, [1], self.device)
        self.assertEqual(wrapper1, wrapper2, "Deepcopied wrappers should be equal.")
        self.assertEqual(hash(wrapper1), hash(wrapper2), "Hashes of equal wrappers should be equal.")
        self.assertNotEqual(wrapper1, wrapper3, "Wrappers with different niches should not be equal.")
        self.assertNotEqual(hash(wrapper1), hash(wrapper3), "Hashes of unequal wrappers should not be equal.")
        model_set = {wrapper1, wrapper2}
        self.assertEqual(len(model_set), 1, "A set should not contain duplicate ModelWrappers.")
        model_set.add(wrapper3)
        self.assertEqual(len(model_set), 2, "A set should be able to contain different ModelWrappers.")

    @patch('src.model_wrapper.ModelWrapper.evaluate_by_class')
    def test_healing_selection_produces_diverse_pairs(self, mock_evaluate_by_class):
        """
        Tests that the HealingMateSelectionStrategy produces a diverse set of
        parent pairs when asked for multiple pairs.
        """
        mock_evaluate_by_class.return_value = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        parent1 = create_mock_wrapper(ModelName.CIFAR10, list(range(10)), self.device, fitness=95.0)
        specialist1 = create_mock_wrapper(ModelName.CIFAR10, [0], self.device, fitness=80.0)
        specialist2 = create_mock_wrapper(ModelName.CIFAR10, [1], self.device, fitness=85.0)
        high_performer = create_mock_wrapper(ModelName.CIFAR10, [9], self.device, fitness=90.0)

        population = [parent1, specialist1, specialist2, high_performer]

        strategy = HealingMateSelectionStrategy()
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.dataset_name = 'CIFAR10'
        config_manager.subset_percentage = 1.0
        config_manager.seed = 42

        num_pairs = 3
        pairs = strategy.select_parent_pairs(population, num_pairs, config_manager=config_manager)

        self.assertEqual(len(pairs), num_pairs, f"Expected {num_pairs} pairs, but got {len(pairs)}.")

        # Check that parent1 is always the first in the pair
        for p1, _ in pairs:
            self.assertIs(p1, parent1)

        # Check that partners are unique and chosen in the correct order
        partners = [p2 for _, p2 in pairs]
        self.assertEqual(len(set(partners)), num_pairs, "The selected partners are not unique.")
        self.assertIs(partners[0], specialist1)
        self.assertIs(partners[1], specialist2)
        self.assertIs(partners[2], high_performer)

    @patch('src.model_wrapper.ModelWrapper.evaluate_by_class')
    def test_healing_selection_fallback_skips_identical_clone(self, mock_evaluate_by_class):
        mock_evaluate_by_class.return_value = [10] * 10
        parent1 = create_mock_wrapper(ModelName.CIFAR10, [0], self.device, fitness=95.0)
        clone_of_parent1 = copy.deepcopy(parent1)
        clone_of_parent1.fitness = 95.0
        expected_parent2 = create_mock_wrapper(ModelName.CIFAR10, [1], self.device, fitness=90.0)

        population = [parent1, clone_of_parent1, expected_parent2]

        strategy = HealingMateSelectionStrategy()
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.dataset_name = 'CIFAR10'
        config_manager.subset_percentage = 1.0
        config_manager.seed = 42

        pairs = strategy.select_parent_pairs(population, 1, config_manager=config_manager)

        self.assertEqual(len(pairs), 1)
        selected_parent1, selected_parent2 = pairs[0]

        self.assertNotEqual(selected_parent1, selected_parent2, "Selected parents should be genetically different.")
        self.assertEqual(selected_parent2, expected_parent2, "The fallback did not select the next-best genetically distinct model.")

    def test_layer_wise_merge_on_resnet_is_not_all_or_nothing(self):
        seed = 42
        parent1 = create_mock_wrapper(ModelName.RESNET, [0], self.device)
        parent2 = create_mock_wrapper(ModelName.RESNET, [1], self.device)
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
        parent1 = create_mock_wrapper(ModelName.CIFAR10, [0], self.device, fitness=90.0)
        parent2 = create_mock_wrapper(ModelName.CIFAR10, [1], self.device, fitness=80.0)
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
        model_wrapper = create_mock_wrapper(ModelName.CIFAR10, [0], self.device)
        from src.evolution import specialize
        from src.config_manager import ConfigManager
        config_manager = ConfigManager('config.yaml')
        config_manager.specialize_epochs = 1
        config_manager.show_progress_bar = True
        specialize(model_wrapper, config_manager)
        mock_tqdm.assert_called_once()
        self.assertTrue(mock_tqdm.return_value.set_postfix.called)
        mock_tqdm.reset_mock()
        config_manager.show_progress_bar = False
        specialize(model_wrapper, config_manager)
        mock_tqdm.assert_not_called()

    def test_sequential_constructive_merge_uses_single_batch_optimization(self):
        parent1 = create_mock_wrapper(ModelName.CIFAR10, [0], self.device, fitness=90.0)
        parent2 = create_mock_wrapper(ModelName.CIFAR10, [1], self.device, fitness=80.0)
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
        model_wrapper_cuda._process_batch.return_value = (torch.randn(1, 10), torch.randint(0, 10, (1,)))

        model_wrapper_cpu = MagicMock()
        model_wrapper_cpu.device = 'cpu'
        model_wrapper_cpu._process_batch.return_value = (torch.randn(1, 10), torch.randint(0, 10, (1,)))

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

    def test_mixed_precision_backward_outside_autocast(self):
        """
        Verifies that scaler.scale().backward() is called *outside* the
        autocast context during mixed-precision training. This is a critical
        check to ensure correct gradient scaling.
        """
        from src.evolution import _run_training_epoch
        from unittest.mock import call

        # 1. Setup mocks
        model_wrapper = MagicMock(device='cuda')
        model_wrapper._process_batch.return_value = (torch.randn(1, 10), torch.randint(0, 10, (1,)))
        optimizer = MagicMock()
        dummy_loader = [(torch.randn(1, 1, 1, 1), torch.randn(1, 1))]
        scaler = MagicMock()
        scaled_loss = MagicMock()
        scaler.scale.return_value = scaled_loss

        # 2. Use a manager to track the call order of context entry/exit and backward()
        manager = MagicMock()
        autocast_context = MagicMock()
        autocast_context.__enter__ = manager.autocast_enter
        autocast_context.__exit__ = manager.autocast_exit
        scaled_loss.backward = manager.backward

        with patch('src.evolution.torch.cuda.amp.autocast', return_value=autocast_context):
            _run_training_epoch(
                model_wrapper,
                optimizer,
                dummy_loader,
                scaler,
                precision='16',
                description='test',
                show_progress_bar=False
            )

        # 3. Assert the correct call order
        expected_calls = [
            call.autocast_enter(),
            call.autocast_exit(None, None, None),
            call.backward()
        ]
        self.assertEqual(manager.mock_calls, expected_calls, "backward() must be called after the autocast context is exited.")


if __name__ == '__main__':
    unittest.main()
