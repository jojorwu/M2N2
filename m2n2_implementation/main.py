"""The main entry point for running the M2N2 evolutionary simulation.

This script orchestrates the entire evolutionary experiment. It loads its
configuration from `config.yaml`, initializes or loads a population of
models, and then runs the main evolutionary loop.
"""
import torch
import os
import glob
import re
import yaml
from .evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation
from .data import get_dataloaders
from .visualization import plot_fitness_history

def main():
    """Runs the main M2N2-inspired evolutionary simulation."""
    # --- 1. Load Configuration from YAML ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # General settings
    model_config = config['model_config']
    precision_config = str(config['precision_config'])
    num_generations = config['num_generations']
    population_size = config['population_size']

    # Evolutionary settings
    merge_strategy = config['merge_strategy']
    mutation_rate = config['mutation_rate']
    initial_mutation_strength = config['initial_mutation_strength']
    mutation_decay_factor = config['mutation_decay_factor']

    # Data settings
    subset_percentage = config['subset_percentage']
    validation_split = config['validation_split']

    # Determine the number of epochs based on model-specific or default settings
    if model_config in config.get('model_specific_epochs', {}):
        specialize_epochs = config['model_specific_epochs'][model_config]['specialize']
        finetune_epochs = config['model_specific_epochs'][model_config]['finetune']
    else:
        specialize_epochs = config['default_epochs']['specialize']
        finetune_epochs = config['default_epochs']['finetune']

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- M2N2 Simplified Implementation ---")
    print(f"Loaded configuration for model: {model_config}")
    print(f"Using device: {DEVICE}\n")

    # --- 2. Create DataLoaders ---
    _, validation_loader, _ = get_dataloaders(
        dataset_name=model_config,
        batch_size=64,
        subset_percentage=subset_percentage,
        validation_split=validation_split
    )

    # --- 3. Initialize or Load Population ---
    print("--- STEP 1: Initializing or Loading Population ---")
    model_dir = "m2n2_implementation/pretrained_models"
    population = []
    niches = [[i] for i in range(population_size)]

    model_files = glob.glob(os.path.join(model_dir, "*.pth"))

    if model_files:
        print(f"Found {len(model_files)} models in {model_dir}. Loading them.")
        for f in model_files:
            match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(f))
            if match:
                niche_classes = [int(n) for n in match.group(1).split('_')]
                fitness = float(match.group(2))
                wrapper = ModelWrapper(model_name=model_config, niche_classes=niche_classes, device=DEVICE)
                wrapper.model.load_state_dict(torch.load(f, map_location=DEVICE))
                wrapper.fitness = fitness
                wrapper.fitness_is_current = True
                population.append(wrapper)
    else:
        print("No pretrained models found. Initializing a new population from scratch.")
        for i in range(population_size):
            population.append(ModelWrapper(model_name=model_config, niche_classes=niches[i], device=DEVICE))

        print("--- Specializing Initial Models ---")
        for model_wrapper in population:
            specialize(model_wrapper, epochs=specialize_epochs, precision=precision_config)
        print("")

    fitness_history = []

    # --- 4. Main Evolutionary Loop ---
    for generation in range(num_generations):
        print(f"\n--- GENERATION {generation + 1}/{num_generations} ---")

        if generation > 0:
            print("--- Specializing Models ---")
            for model_wrapper in population:
                if model_wrapper.niche_classes != list(range(10)):
                    specialize(model_wrapper, epochs=specialize_epochs, precision=precision_config)
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
            child = merge(parent1, parent2, strategy=merge_strategy, validation_loader=validation_loader)
            child = mutate(
                child,
                generation=generation,
                mutation_rate=mutation_rate,
                initial_mutation_strength=initial_mutation_strength,
                decay_factor=mutation_decay_factor
            )
            finetune(child, epochs=finetune_epochs, precision=precision_config)
            population = create_next_generation(population, child, population_size)
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
        files = glob.glob(os.path.join(model_dir, "*.pth"))
        for f in files:
            os.remove(f)

    for i, model_wrapper in enumerate(population):
        niche_str = "_".join(map(str, model_wrapper.niche_classes))
        model_path = os.path.join(model_dir, f"model_niche_{niche_str}_fitness_{model_wrapper.fitness:.2f}.pth")
        torch.save(model_wrapper.model.state_dict(), model_path)
        print(f"  - Saved model to {model_path}")

if __name__ == '__main__':
    main()