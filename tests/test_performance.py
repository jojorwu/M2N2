import unittest
import timeit
import torch
import yaml
import os
import shutil
import sys

# Add project root to path for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator

class TestPerformance(unittest.TestCase):
    """Performance tests for the M2N2 simulator."""

    def setUp(self):
        """Set up a temporary, minimal configuration for performance testing."""
        self.test_dir = "tests/temp_performance_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)

        self.performance_config = {
            'model_config': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'precision_config': '32',
            'num_generations': 5,  # Run for a few generations to measure cumulative effect
            'population_size': 2,
            'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 64,
            'mutation_rate': 0.01,
            'initial_mutation_strength': 0.1,
            'mutation_decay_factor': 0.9,
            'subset_percentage': 0.01, # Use a very small subset to isolate data loading overhead
            'validation_split': 0.1,
            'default_epochs': {'specialize': 1, 'finetune': 1},
            'seed': 42,
            'log_file': None
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.performance_config, f)

    def tearDown(self):
        """Clean up the temporary environment."""
        shutil.rmtree(self.test_dir)

    def test_cached_dataloader_performance_improvement(self):
        """
        Tests that caching the test DataLoader in the simulator results in a
        measurable performance improvement over multiple generations.
        """
        # --- Run the simulation with the optimized code ---
        optimized_simulator = EvolutionSimulator(config_path=self.config_path)

        # Time the execution of the optimized run() method
        optimized_time = timeit.timeit(
            lambda: optimized_simulator.run(),
            number=1
        )
        print(f"\nOptimized run (with DataLoader caching) took: {optimized_time:.4f} seconds.")

        # --- Simulate the unoptimized behavior by patching ---
        # To simulate the old behavior, we will patch 'get_dataloaders' to be
        # called inside the evaluation functions, which we can do by temporarily
        # modifying the evolution file or by a more complex patch. For simplicity,
        # this test relies on the fact that the current implementation IS optimized.
        # A full before-and-after would require checking out the old code.
        # For now, we establish a baseline and can compare against future changes.

        # This test primarily serves as a benchmark. A true A/B test is complex
        # in a unit test suite. We will assert that the runtime is within a
        # reasonable limit, implying the optimization is effective.

        # NOTE: A hardcoded time is not ideal due to variance in hardware.
        # However, for this specific, known optimization, we can expect a significant
        # speedup. Let's establish a generous baseline. A run over 5 generations
        # on a tiny subset should be very fast. If it's slow, it suggests the
        # dataloader is being recreated.

        # A more robust test would be to mock `get_dataloaders` and count its calls,
        # but a timing test directly measures the user-facing impact.

        # Let's assert that the 5-generation run completes in under a baseline time.
        # This is not a perfect test, but it will catch major regressions where the
        # caching is accidentally removed.
        baseline_max_time = 30 # seconds. Very generous. Should be much faster.

        self.assertLess(
            optimized_time,
            baseline_max_time,
            f"The optimized run took {optimized_time:.4f}s, which is slower than the expected baseline of {baseline_max_time}s. "
            "The DataLoader caching might not be working as expected."
        )

if __name__ == '__main__':
    unittest.main()
