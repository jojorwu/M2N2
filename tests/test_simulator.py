import unittest
from unittest.mock import patch
import os
import shutil
import torch
import yaml
import sys
import pytest

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model import CifarCNN
from src.evolution import ModelWrapper

class TestSimulator(unittest.TestCase):
    """Unit tests for the EvolutionSimulator's logic."""

    def setUp(self):
        """Set up a temporary environment for the simulator test."""
        self.test_dir = "tests/temp_simulator_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)

        # Base config that can be modified by each test
        self.base_config = {
            'model_config': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'model_dir': os.path.join(self.test_dir, 'models'),
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 1,
            'merge_strategy': 'average',
            'mutation_rate': 0.0,
            'initial_mutation_strength': 0.0,
            'mutation_decay_factor': 1.0,
            'subset_percentage': 0.01,
            'validation_split': 0.1,
            'default_epochs': {'specialize': 0, 'finetune': 0},
            'dataset_configs': {
                'CIFAR10': {'num_classes': 10},
                'MNIST': {'num_classes': 10},
                'LLM': {'num_classes': 77},
                'IMAGENET': {'num_classes': 1000}
            }
        }

    def tearDown(self):
        """Clean up the temporary environment after the test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('src.simulator.glob.glob')
    def test_loaded_model_fitness_is_marked_as_stale(self, mock_glob):
        """
        Tests that a model loaded from a file has its `fitness_is_current`
        flag set to False, forcing a re-evaluation on the new dataset.
        """
        model_dir = os.path.join(self.test_dir, "pretrained_models")
        os.makedirs(model_dir, exist_ok=True)
        dummy_model_path = os.path.join(model_dir, "model_niche_0_fitness_99.9.pth")
        torch.save(CifarCNN().state_dict(), dummy_model_path)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        mock_glob.return_value = [dummy_model_path]
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertGreater(len(simulator.population), 0, "Simulator failed to load any models.")
        self.assertFalse(simulator.population[0].fitness_is_current)

    @pytest.mark.slow
    def test_initial_specialization_is_deterministic_with_seed(self):
        """
        Tests that when a seed is provided, two simulators created with the
        same config will have identical initial populations after specialization.
        """
        config = self.base_config.copy()
        config['default_epochs']['specialize'] = 1
        config['population_size'] = 2
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        sim1 = EvolutionSimulator(config_path=self.config_path, seed=42)
        model1_params = [p.clone() for p in sim1.population[0].model.parameters()]

        sim2 = EvolutionSimulator(config_path=self.config_path, seed=42)
        model2_params = [p.clone() for p in sim2.population[0].model.parameters()]

        self.assertEqual(len(model1_params), len(model2_params))
        for p1, p2 in zip(model1_params, model2_params):
            self.assertTrue(torch.equal(p1, p2),
                            "Model weights are not identical between two simulators with the same seed.")

    @patch('src.simulator.ProcessPoolExecutor')
    def test_parallel_specialization_logic_is_correct(self, MockProcessPoolExecutor):
        """
        Verifies the logic of parallel specialization by mocking the executor.
        """
        mock_executor_instance = MockProcessPoolExecutor.return_value.__enter__.return_value
        dummy_state_dict = CifarCNN(num_classes=10).state_dict()
        mock_executor_instance.map.return_value = [dummy_state_dict] * 4

        config = self.base_config.copy()
        config['population_size'] = 4
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        simulator = EvolutionSimulator(config_path=self.config_path, seed=1337)

        self.assertTrue(MockProcessPoolExecutor.called)
        self.assertEqual(len(simulator.population), config['population_size'])
        self.assertEqual(
            list(simulator.population[0].model.state_dict().keys()),
            list(dummy_state_dict.keys())
        )

    @patch('src.simulator.ProcessPoolExecutor')
    @patch('src.simulator.EvolutionSimulator._initialize_population')
    def test_parallel_evaluation_logic_is_correct(self, mock_init_pop, MockProcessPoolExecutor):
        """
        Verifies the logic of parallel evaluation by mocking the executor.
        """
        mock_executor_instance = MockProcessPoolExecutor.return_value.__enter__.return_value
        mock_executor_instance.map.return_value = [99.9] * 4
        mock_init_pop.return_value = None

        config = self.base_config.copy()
        config['population_size'] = 4
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        simulator = EvolutionSimulator(config_path=self.config_path, seed=1337)
        simulator.population = [ModelWrapper(model_name='CIFAR10', niche_classes=[i], num_classes=10, device=simulator.device) for i in range(4)]
        for model_wrapper in simulator.population:
            model_wrapper.fitness_is_current = False

        with patch('src.simulator.plot_fitness_history'):
            simulator.run()

        self.assertTrue(MockProcessPoolExecutor.called)
        final_fitnesses = [m.fitness for m in simulator.population]
        self.assertEqual(final_fitnesses, [99.9] * config['population_size'])

if __name__ == '__main__':
    unittest.main()