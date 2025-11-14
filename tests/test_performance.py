"""Performance tests for the M2N2 simulation.

This module contains tests that are specifically designed to measure and
assert the performance of computationally intensive parts of the codebase.
These tests help prevent regressions that could slow down the simulation.
"""

import unittest
import time
import torch
from src.model_wrapper import ModelWrapper
from src.evolution import mutate, evaluate_by_class, merge
from src.enums import ModelName
from src.simulator import EvolutionSimulator
import logging

# Disable logging for performance tests to avoid I/O overhead
logging.disable(logging.CRITICAL)

class TestPerformance(unittest.TestCase):
    """Contains performance-related test cases."""

    def test_mutate_performance(self):
        """
        Tests that the mutate function executes within an acceptable time frame.

        This test creates a reasonably large model and calls the mutate
        function on it multiple times, asserting that the total execution
        time is below a generous threshold. A failure in this test indicates
        a significant performance regression in the mutation logic.

        The threshold is set to 0.5 seconds, which is substantially higher
        than the expected execution time of the optimized function (which should
        be in the low milliseconds), but low enough to catch the unoptimized
        version, which would take several seconds.
        """
        # 1. Setup a model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use a model with a significant number of parameters for a realistic test
        model_wrapper = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=list(range(10)),
            device=device,
            num_classes=10
        )
        # Ensure model is on the correct device
        model_wrapper.model.to(device)


        # 2. Time the mutate function over several runs
        num_runs = 25
        start_time = time.time()
        for i in range(num_runs):
            # The generation number affects the mutation strength, so vary it
            mutate(model_wrapper, generation=i)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / num_runs

        # 3. Assert performance
        # This threshold is generous. The unoptimized version takes several
        # seconds, while the optimized version should be in the low milliseconds.
        # This test is designed to catch major regressions.
        performance_threshold = 1.5  # seconds
        print(f"Execution of {num_runs} mutations took {total_time:.4f} seconds (avg: {avg_time:.4f}s).")
        self.assertLess(
            total_time,
            performance_threshold,
            f"Mutation function is too slow. "
            f"Execution of {num_runs} runs took {total_time:.4f} seconds, "
            f"which is over the threshold of {performance_threshold} seconds."
        )

    def test_evaluate_by_class_performance(self):
        """
        Tests that the evaluate_by_class function executes efficiently.

        This test evaluates the performance of the class-wise evaluation
        function, which is critical for the mate selection process. A failure
        here indicates a performance regression that could slow down each
        generation of the simulation.

        The threshold is set to be generous, easily passing the vectorized
        implementation while catching the slow, loop-based version.
        """
        # 1. Setup a model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_wrapper = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=list(range(10)),
            device=device,
            num_classes=10
        )
        model_wrapper.model.to(device)

        # 2. Time the function over several runs
        num_runs = 5
        start_time = time.time()
        for _ in range(num_runs):
            evaluate_by_class(
                model_wrapper,
                dataset_name='CIFAR10',
                subset_percentage=0.1  # Use a subset to keep test time reasonable
            )
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_runs

        # 3. Assert performance
        performance_threshold = 20.0  # seconds
        print(f"Execution of {num_runs} class evaluations took {total_time:.4f} seconds (avg: {avg_time:.4f}s).")
        self.assertLess(
            total_time,
            performance_threshold,
            f"Class evaluation function is too slow. "
            f"Execution of {num_runs} runs took {total_time:.4f} seconds, "
            f"exceeding the {performance_threshold}s threshold."
        )


    def test_sequential_constructive_merge_performance(self):
        """
        Tests the performance of the 'sequential_constructive' merge strategy.

        The original implementation of this strategy was very slow due to
        repeatedly evaluating the model. This test ensures that the optimized
        heuristic runs within a very generous time limit, preventing a
        regression to the slow algorithm.
        """
        # 1. Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        parent1 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=list(range(10)),
            device=device,
            num_classes=10
        )
        parent1.fitness = 80.0
        parent2 = ModelWrapper(
            model_name=ModelName.CIFAR10,
            niche_classes=list(range(10)),
            device=device,
            num_classes=10
        )
        parent2.fitness = 70.0

        # Create a dummy validation loader with one batch
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
        )
        validation_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16)

        # 2. Time the merge function
        start_time = time.time()
        merge(
            parent1,
            parent2,
            strategy='sequential_constructive',
            validation_loader=validation_loader
        )
        end_time = time.time()
        total_time = end_time - start_time

        # 3. Assert performance
        # The original version could take minutes. The optimized version
        # should take a fraction of a second.
        performance_threshold = 2.0  # seconds
        print(f"Execution of sequential_constructive merge took {total_time:.4f} seconds.")
        self.assertLess(
            total_time,
            performance_threshold,
            f"Sequential constructive merge is too slow. "
            f"Execution took {total_time:.4f}s, exceeding the {performance_threshold}s threshold."
        )


    def test_single_generation_performance(self):
        """
        Tests the performance of a single generation run.
        """
        import yaml
        import os

        # Create a temporary config file for this test
        test_config = {
            'model_config': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 2,
            'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'mutation_rate': 0.01,
            'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 0.9,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'subset_percentage': 0.01,
            'validation_split': 0.1,
            'batch_size': 32,
            'default_epochs': {'specialize': 1, 'finetune': 1}
        }
        config_path = 'temp_test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)

        # 1. Setup the simulator with the temporary config
        simulator = EvolutionSimulator(config_path=config_path)

        # 2. Time the execution of one generation
        start_time = time.time()
        simulator.run_one_generation()
        end_time = time.time()
        total_time = end_time - start_time

        # 3. Assert performance
        performance_threshold = 60.0  # seconds
        print(f"Execution of one generation took {total_time:.4f} seconds.")
        self.assertLess(
            total_time,
            performance_threshold,
            f"A single generation is too slow. "
            f"Execution took {total_time:.4f}s, exceeding the {performance_threshold}s threshold."
        )

        # Cleanup the temporary config file
        os.remove(config_path)


if __name__ == '__main__':
    unittest.main()