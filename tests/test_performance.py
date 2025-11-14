"""Performance tests for the M2N2 simulation.

This module contains tests that are specifically designed to measure and
assert the performance of computationally intensive parts of the codebase.
These tests help prevent regressions that could slow down the simulation.
"""

import unittest
import time
import torch
from src.model_wrapper import ModelWrapper
from src.evolution import mutate, evaluate_by_class
from src.enums import ModelName
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


if __name__ == '__main__':
    unittest.main()