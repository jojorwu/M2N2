import torch
from evolution import ModelWrapper, specialize, evaluate, select_mates, merge, mutate, finetune

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

    POPULATION_SIZE_PER_NICHE = 3  # Keep it small for a quick run
    NICHE_1_DIGITS = list(range(5))   # Digits 0-4
    NICHE_2_DIGITS = list(range(5, 10)) # Digits 5-9

    # --- 2. Initialize Population ---
    print("--- STEP 1: Initializing Population ---")
    population = []
    for _ in range(POPULATION_SIZE_PER_NICHE):
        population.append(ModelWrapper(niche_digits=NICHE_1_DIGITS, device=DEVICE))
        population.append(ModelWrapper(niche_digits=NICHE_2_DIGITS, device=DEVICE))
    print(f"Initialized population of {len(population)} models ({POPULATION_SIZE_PER_NICHE} per niche).\n")

    # --- 3. Specialization Phase ---
    print("--- STEP 2: Specialization (Training on Niches) ---")
    for model_wrapper in population:
        # We train for a few epochs to make them specialists
        specialize(model_wrapper, epochs=2)
    print("")

    # --- 4. Evaluation Phase ---
    print("--- STEP 3: Evaluating All Specialists ---")
    for model_wrapper in population:
        evaluate(model_wrapper)
    print("")

    # --- 5. Mating and Evolution ---
    print("--- STEP 4: Mating, Merging, and Mutation ---")
    # Select the fittest model from each niche to act as parents
    parent1, parent2 = select_mates(population)

    # Merge the two parents to create a new, more generalist child
    child = merge(parent1, parent2)

    # Apply mutation to the child to introduce new "genetic" material
    child = mutate(child, mutation_rate=0.05, mutation_strength=0.1)

    # Fine-tune the child on the full dataset to help it learn from the merge
    finetune(child, epochs=1)
    print("")

    # --- 6. Final Evaluation ---
    print("--- STEP 5: Final Evaluation of Evolved Child ---")
    child_fitness = evaluate(child)

    print("\n\n--- EXPERIMENT SUMMARY ---")
    print(f"Best specialist for {parent1.niche_digits} had a general accuracy of: {parent1.fitness:.2f}%")
    print(f"Best specialist for {parent2.niche_digits} had a general accuracy of: {parent2.fitness:.2f}%")
    print("--------------------------------------------------")
    print(f"The evolved child model achieved a final accuracy of: {child_fitness:.2f}%")
    print("--------------------------------------------------")

    if child_fitness > parent1.fitness and child_fitness > parent2.fitness:
        print("\nCONCLUSION: Success! The evolved child outperformed both of its specialist parents.")
    else:
        print("\nCONCLUSION: The child did not outperform both parents. This can happen due to the stochastic nature of training and mutation. Try running the experiment again!")

if __name__ == '__main__':
    main()
