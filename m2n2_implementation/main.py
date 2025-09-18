import torch
from evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune, create_next_generation

def main():
    """Runs the main M2N2-inspired evolutionary experiment.

    This script orchestrates a multi-generational simulation of model evolution.
    The process for each generation is as follows:
    1.  **Initialization**: A population of specialist models is created,
        each assigned to a specific data niche (one for each CIFAR-10 class).
    2.  **Specialization**: Each specialist model is trained on the data
        from its assigned niche. This step is skipped for generalist models
        in subsequent generations.
    3.  **Evaluation**: All models in the population are evaluated on the
        full, general dataset to determine their fitness score.
    4.  **Mating**: An advanced selection strategy picks two parents: the
        fittest model overall (Parent 1) and the specialist for that
        model's weakest class (Parent 2).
    5.  **Crossover & Mutation**: The parents are merged to create a new
        child model, which is then mutated to introduce genetic diversity
        and fine-tuned on the general dataset.
    6.  **Selection**: The new child competes with the existing population.
        The fittest individuals are selected to form the next generation
        (elitism).

    The simulation runs for a configured number of generations, printing a
    summary of fitness scores at the end.
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
