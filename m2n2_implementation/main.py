"""The main entry point for running the M2N2 evolutionary simulation.

This script orchestrates the entire evolutionary experiment, from initializing
a population of models to running the generational loop of evaluation,
selection, merging, and mutation. It also handles the loading and saving of
model populations to allow for iterative experiments.
"""
import torch
import os
import glob
import re
from evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation

def main():
    """Runs the main M2N2-inspired evolutionary experiment.

    This function orchestrates a multi-generational simulation of model
    evolution. The process is as follows:

    1.  **Configuration**: Sets up the device (CPU/GPU), population size,
        niche configuration, and number of generations.
    2.  **Initialization/Loading**: Checks for a pre-existing population of
        models in `pretrained_models/`. If found, it loads them. Otherwise,
        it initializes a new population of specialist models from scratch
        and trains them on their respective niches.
    3.  **Evolutionary Loop**: For each generation, it performs:
        a. **Specialization**: Specialist models (not generalist children)
           are trained on their niche data.
        b. **Evaluation**: All models are evaluated on the full test set to
           determine their fitness.
        c. **Mating**: The advanced `select_mates` strategy is used to pick
           two parents for crossover.
        d. **Crossover & Mutation**: The parents are merged into a child,
           which is then mutated and fine-tuned.
        e. **Selection**: The new child competes with the old population,
           and the fittest individuals survive to the next generation.
    4.  **Summary**: After all generations are complete, it prints a summary
        of the fitness history.
    5.  **Save Population**: The final, evolved population is saved to the
        `pretrained_models/` directory, overwriting any previous run.
    """
    # --- 1. Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- M2N2 Simplified Implementation ---")
    print(f"Using device: {DEVICE}\n")

    POPULATION_SIZE = 10 # One specialist for each of the 10 classes
    NICHES = [[i] for i in range(10)] # e.g., [[0], [1], [2], ...]
    NUM_GENERATIONS = 2 # Keep low for testing

    # --- 2. Initialize or Load Population ---
    print("--- STEP 1: Initializing or Loading Population ---")
    model_dir = "m2n2_implementation/pretrained_models"
    population = []

    # Check if there are models to load
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))

    if model_files:
        print(f"Found {len(model_files)} models in {model_dir}. Loading them.")
        for f in model_files:
            # Extract niche and fitness from filename, e.g., "model_niche_0_fitness_10.00.pth"
            match = re.search(r'model_niche_([\d_]+)_fitness_([\d\.]+)\.pth', os.path.basename(f))
            if match:
                niche_str = match.group(1)
                fitness_str = match.group(2)
                niche_classes = [int(n) for n in niche_str.split('_')]
                fitness = float(fitness_str)

                # Create a wrapper, load the state dict, and set the fitness
                wrapper = ModelWrapper(niche_classes=niche_classes, device=DEVICE)
                wrapper.model.load_state_dict(torch.load(f, map_location=DEVICE))
                wrapper.fitness = fitness
                population.append(wrapper)
                print(f"  - Loaded model for niche {niche_classes} (Fitness: {fitness:.2f}) from {os.path.basename(f)}")
            else:
                print(f"  - WARNING: Could not parse niche from filename: {os.path.basename(f)}")

        # If we loaded models, we might need to top-up the population if there aren't enough
        # For simplicity, this implementation assumes a full set of models exists if any exist.

    else:
        print("No pretrained models found. Initializing a new population from scratch.")
        for i in range(POPULATION_SIZE):
            niche = NICHES[i]
            population.append(ModelWrapper(niche_classes=niche, device=DEVICE))
        print(f"Initialized population of {len(population)} models, one for each class.\n")

        # The initial population needs to be specialized
        print("--- Specializing Initial Models ---")
        for model_wrapper in population:
            specialize(model_wrapper, epochs=1) # Initial specialization
        print("")

    fitness_history = []

    # --- Main Evolutionary Loop ---
    for generation in range(NUM_GENERATIONS):
        print(f"\n--- GENERATION {generation + 1}/{NUM_GENERATIONS} ---")

        # --- 3. Specialization Phase (if needed) ---
        if generation > 0:
            print("--- Specializing Models ---")
            for model_wrapper in population:
                # Only specialize the specialists, not the generalist children
                if model_wrapper.niche_classes != list(range(10)):
                    specialize(model_wrapper, epochs=1)
            print("")

        # --- 4. Evaluation Phase ---
        print("--- Evaluating Population ---")
        for model_wrapper in population:
            evaluate(model_wrapper)

        best_fitness = max([m.fitness for m in population])
        avg_fitness = sum([m.fitness for m in population]) / len(population)
        fitness_history.append((best_fitness, avg_fitness))
        print(f"\nGeneration {generation + 1} Stats: Best Fitness = {best_fitness:.2f}%, Avg Fitness = {avg_fitness:.2f}%\n")

        # --- 5. Mating and Evolution ---
        print("--- Mating and Evolution ---")
        parent1, parent2 = select_mates(population)

        if parent1 and parent2:
            child = merge(parent1, parent2, strategy='fitness_weighted')
            child = mutate(child, mutation_rate=0.05, mutation_strength=0.1)
            finetune(child, epochs=1)

            # --- 6. Create Next Generation ---
            population = create_next_generation(population, child, POPULATION_SIZE)
        else:
            # If no suitable parents were found, the population carries over without a new child
            print("Population will carry over to the next generation without changes.")

    print("\n\n--- EXPERIMENT SUMMARY ---")
    print("Fitness history (Best, Average):")
    for i, (best, avg) in enumerate(fitness_history):
        print(f"  - Generation {i+1}: Best={best:.2f}%, Avg={avg:.2f}%")

    final_best_model = max(population, key=lambda m: m.fitness)
    print(f"\nFinal best model achieved an accuracy of {final_best_model.fitness:.2f}%")

    # --- 7. Save Final Population ---
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
