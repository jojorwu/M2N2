import torch
from evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation

def main():
    """
    The main script to run the M2N2-inspired evolutionary experiment.
    This script simulates one generation of evolution:
    1. Initialize a population of models, each assigned to a data niche.
    2. Train (specialize) each model on its niche.
    3. Evaluate all models on the full dataset to find their fitness.
    4. Select the best models from complementary niches to be parents.
    5. Merge (mate) the parents to create a child.
    6. Mutate the child to introduce diversity.
    7. Evaluate the final child and compare its performance to its parents.
    """
    # --- 1. Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- M2N2 Simplified Implementation ---")
    print(f"Using device: {DEVICE}\n")

    POPULATION_SIZE = 10 # One specialist for each of the 10 classes
    NICHES = [[i] for i in range(10)] # e.g., [[0], [1], [2], ...]
    NUM_GENERATIONS = 2 # Keep low for testing

    # --- 2. Initialize Population ---
    print("--- STEP 1: Initializing Population ---")
    population = []
    for i in range(POPULATION_SIZE):
        niche = NICHES[i]
        population.append(ModelWrapper(niche_classes=niche, device=DEVICE))
    print(f"Initialized population of {len(population)} models, one for each class.\n")

    fitness_history = []

    # --- Main Evolutionary Loop ---
    for generation in range(NUM_GENERATIONS):
        print(f"\n--- GENERATION {generation + 1}/{NUM_GENERATIONS} ---")

        # --- 3. Specialization Phase ---
        print("--- Specializing Models ---")
        for model_wrapper in population:
            if model_wrapper.niche_classes != list(range(10)): # Don't re-specialize generalists
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

if __name__ == '__main__':
    main()
