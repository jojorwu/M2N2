"""Handles the visualization of experiment results.

This module provides functions for plotting and saving the results of the
evolutionary simulation, such as the fitness history over generations.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_fitness_history(fitness_history, output_path):
    """Plots and saves the fitness history of the population.

    This function generates a line plot showing the best and average
    fitness scores for each generation and saves it to a file.

    Args:
        fitness_history (list[tuple[float, float]]): A list of tuples,
            where each tuple contains the (best_fitness, avg_fitness)
            for a generation.
        output_path (str): The file path where the plot image will be
            saved (e.g., 'fitness_history.png').
    """
    generations = range(1, len(fitness_history) + 1)
    best_fitness = [f[0] for f in fitness_history]
    avg_fitness = [f[1] for f in fitness_history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'b-', label='Best Fitness')
    plt.plot(generations, avg_fitness, 'r--', label='Average Fitness')

    plt.title('Fitness History Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Accuracy %)')
    plt.legend()
    plt.grid(True)

    # BUG FIX: Use MaxNLocator to ensure the x-axis has a reasonable number
    # of integer ticks, preventing overcrowding on long runs.
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    plt.savefig(output_path)
    print(f"\nFitness history plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    # Example usage:
    # Create some dummy data
    dummy_history = [(25.0, 15.0), (35.5, 22.3), (45.8, 30.1), (55.2, 40.5), (60.0, 48.9)]
    # Define an output path
    dummy_output_path = 'sample_fitness_plot.png'

    # Generate and save the plot
    plot_fitness_history(dummy_history, dummy_output_path)

    print(f"A sample plot has been generated and saved to '{dummy_output_path}'.")