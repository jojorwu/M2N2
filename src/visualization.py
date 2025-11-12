"""
Handles the visualization of experiment results.

This module provides functions for plotting and saving the results of the
evolutionary simulation, such as the fitness history over generations.
It uses Matplotlib to generate static plots.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple

def plot_fitness_history(fitness_history: List[Tuple[float, float]], output_path: str) -> None:
    """
    Plots and saves the fitness history of the population over generations.

    This function generates a line plot showing the best and average fitness
    scores for each generation. The x-axis represents the generation number,
    and the y-axis represents the fitness (accuracy percentage). The plot is
    saved to a specified file.

    Args:
        fitness_history (List[Tuple[float, float]]): A list of tuples, where
            each tuple contains the (best_fitness, average_fitness) for a
            generation.
        output_path (str): The file path where the plot image will be saved
            (e.g., 'fitness_history.png').
    """
    generations = range(1, len(fitness_history) + 1)
    best_fitness = [f[0] for f in fitness_history]
    avg_fitness = [f[1] for f in fitness_history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'b-o', label='Best Fitness')
    plt.plot(generations, avg_fitness, 'r--x', label='Average Fitness')

    plt.title('Fitness History Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Accuracy %)')
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune=None))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    """
    Example script to demonstrate the plotting functionality.

    When this script is run directly, it generates a sample fitness plot
    from dummy data and saves it, serving as a quick visual test.
    """
    dummy_history = [(25.0, 15.0), (35.5, 22.3), (45.8, 30.1), (55.2, 40.5), (60.0, 48.9)]
    dummy_output_path = 'sample_fitness_plot.png'
    plot_fitness_history(dummy_history, dummy_output_path)
    print(f"A sample plot has been generated and saved to '{dummy_output_path}'.")
