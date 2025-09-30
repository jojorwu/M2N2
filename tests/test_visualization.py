import unittest
from unittest.mock import patch
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path to allow for package-like imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2n2_implementation.visualization import plot_fitness_history

class TestVisualization(unittest.TestCase):
    """Tests for the visualization module."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_many_generations_has_reasonable_ticks(self, mock_close, mock_savefig):
        """
        Tests that the fixed plotting logic uses MaxNLocator to create a
        reasonable number of ticks for a large number of generations.
        """
        # Arrange
        num_generations = 50
        max_expected_ticks = 15 # A reasonable upper limit for readability
        dummy_history = [(i, i - 5) for i in range(1, num_generations + 1)]

        # Act
        # Call the function. The mocks will prevent it from saving or closing the plot.
        plot_fitness_history(dummy_history, "dummy_path.png")

        # Get the current figure and axes to inspect them
        fig = plt.gcf()
        ax = fig.gca()

        num_ticks = len(ax.get_xticks())

        # Clean up the plot object after inspection
        plt.close(fig)

        # Assert
        # This confirms the fix: the number of ticks is now managed by MaxNLocator
        # and should be well below the total number of generations.
        self.assertLessEqual(
            num_ticks,
            max_expected_ticks,
            f"Expected a maximum of {max_expected_ticks} ticks, but got {num_ticks}. "
            "The plot may be overcrowded and unreadable."
        )

if __name__ == '__main__':
    unittest.main()