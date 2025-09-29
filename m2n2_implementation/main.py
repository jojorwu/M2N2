"""The main entry point for running the M2N2 evolutionary simulation.

This script orchestrates the entire evolutionary experiment. It sets up the
configuration, initializes or loads a population of models, and then runs the
main evolutionary loop for a specified number of generations. Finally, it
summarizes the results and saves the evolved population.

This script is designed to be run directly from the command line.
"""
import torch
import os
import glob
import re
from .evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation
from .data import get_dataloaders
from .visualization import plot_fitness_history

def main():
    """Runs the main M2N2-inspired evolutionary simulation.

    This function serves as the primary driver for the experiment,
    orchestrating a multi-generational simulation of model evolution. The
    entire process is logged to the console.
    """
    # --- 1. Configuration ---
    MODEL_CONFIG = 'RESNET' # Options: 'CIFAR10', 'MNIST', 'LLM', 'RESNET'
    PRECISION_CONFIG = '32' # Options: '16' (mixed), '32' (standard), '64' (double)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- M2N2 Simplified Implementation ---")
    print(f"Using model config: {MODEL_CONFIG}")
    print(f"Using device: {DEVICE}\n")

    # --- Experiment-specific Configuration ---
    if MODEL_CONFIG == 'LLM':
        POPULATION_SIZE = 10
        NICHES = [[i] for i in range(POPULATION_SIZE)]
        SPECIALIZE_EPOCHS = 1
        FINETUNE_EPOCHS = 1
    else: # Default to CIFAR10/MNIST/ResNet settings
        POPULATION_SIZE = 10
        NICHES = [[i] for i in range(POPULATION_SIZE)]
        SPECIALIZE_EPOCHS = 2
        FINETUNE_EPOCHS = 3

    NUM_GENERATIONS = 2 # Keep low for testing

    # --- 2. Create DataLoaders ---
    # Create a validation set for the crossover process to use for fast evaluation
    _, validation_loader, _ = get_dataloaders(
        dataset_name=MODEL_CONFIG,
        batch_size=64, # A standard batch size is fine
        subset_percentage=0.1 # Use a small subset for speed
    )

    # --- 3. Initialize or Load Population ---
    print("--- STEP 1: Initializing or Loading Population ---")
    model_dir = "m2n2_implementation/pretrained_models"
    population = []

    model_files = glob.glob(os.path.join(model_dir, "*.pth"))

    if model_files:
        print(f"Found {len(model_files)} models in {model_dir}. Loading them.")
        for f in model_files:
            match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(f))
            if match:
                niche_classes = [int(n) for n in match.group(1).split('_')]
                fitness = float(match.group(2))
                wrapper = ModelWrapper(model_name=MODEL_CONFIG, niche_classes=niche_classes, device=DEVICE)
                wrapper.model.load_state_dict(torch.load(f, map_location=DEVICE))
                wrapper.fitness = fitness
                wrapper.fitness_is_current = True # Fitness from a saved file is considered current
                population.append(wrapper)
    else:
        print("No pretrained models found. Initializing a new population from scratch.")
        for i in range(POPULATION_SIZE):
            population.append(ModelWrapper(model_name=MODEL_CONFIG, niche_classes=NICHES[i], device=DEVICE))

        print("--- Specializing Initial Models ---")
        for model_wrapper in population:
            specialize(model_wrapper, epochs=SPECIALIZE_EPOCHS, precision=PRECISION_CONFIG)
        print("")

    fitness_history = []

    # --- 4. Main Evolutionary Loop ---
    for generation in range(NUM_GENERATIONS):
        print(f"\n--- GENERATION {generation + 1}/{NUM_GENERATIONS} ---")

        if generation > 0:
            print("--- Specializing Models ---")
            for model_wrapper in population:
                if model_wrapper.niche_classes != list(range(10)):
                    specialize(model_wrapper, epochs=SPECIALIZE_EPOCHS, precision=PRECISION_CONFIG)
            print("")

        print("--- Evaluating Population on Test Set ---")
        for model_wrapper in population:
            evaluate(model_wrapper)

        best_fitness = max([m.fitness for m in population])
        avg_fitness = sum([m.fitness for m in population]) / len(population)
        fitness_history.append((best_fitness, avg_fitness))
        print(f"\nGeneration {generation + 1} Stats: Best Fitness = {best_fitness:.2f}%, Avg Fitness = {avg_fitness:.2f}%\n")

        print("--- Mating and Evolution ---")
        parent1, parent2 = select_mates(population)

        if parent1 and parent2:
            child = merge(parent1, parent2, strategy='sequential_constructive', validation_loader=validation_loader)
            child = mutate(child, generation=generation, mutation_rate=0.05)
            finetune(child, epochs=FINETUNE_EPOCHS, precision=PRECISION_CONFIG)
            population = create_next_generation(population, child, POPULATION_SIZE)
        else:
            print("Population will carry over to the next generation without changes.")

    print("\n\n--- EXPERIMENT SUMMARY ---")
    print("Fitness history (Best, Average):")
    for i, (best, avg) in enumerate(fitness_history):
        print(f"  - Generation {i+1}: Best={best:.2f}%, Avg={avg:.2f}%")

    final_best_model = max(population, key=lambda m: m.fitness)
    print(f"\nFinal best model achieved an accuracy of {final_best_model.fitness:.2f}%")

    # --- 7. Visualize and Save ---
    plot_fitness_history(fitness_history, 'fitness_history.png')

    print("\n--- Saving final population to pretrained_models/ ---")
    model_dir = "m2n2_implementation/pretrained_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        # Clear out old models before saving the new generation
        print("Clearing stale models from previous run...")
        files = glob.glob(os.path.join(model_dir, "*.pth"))
        for f in files:
            os.remove(f)

    for i, model_wrapper in enumerate(population):
        # Niche is a list of integers. Create a string like "1_2_3"
        niche_str = "_".join(map(str, model_wrapper.niche_classes))
        model_path = os.path.join(model_dir, f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth")
        torch.save(model_wrapper.model.state_dict(), model_path)
        print(f"  - Saved model to {model_path}")


if __name__ == '__main__':
    main()