# M2N2: A Simplified Implementation (CIFAR-10 Version)

This project is a simplified, educational implementation of the concepts from Sakana AI's research paper on M2N2 (Model Merging of Natural Niches). It demonstrates how a population of neural networks can be evolved through specialization, intelligent mating, and merging to produce a more capable, generalist model.

This version of the project runs on the **CIFAR-10 dataset**.

## Core Concepts Demonstrated

- **Specialization (Resource Competition):** We create a population of CNN models and train them on different "niches" of the CIFAR-10 dataset: "animals" vs. "vehicles".
- **Intelligent Mating:** The best-performing model from each niche is selected to act as a "parent".
- **Fitness-Weighted Merging (Crossover):** The parent models are merged into a "child" model. This implementation uses a 'fitness_weighted' strategy, where the contribution of each parent to the child's weights is proportional to its fitness score.
- **Mutation & Fine-tuning:** The child is mutated and then fine-tuned on the full dataset to help it learn how to integrate its parents' knowledge.

## Multi-Generational Evolution

This implementation supports a multi-generational evolutionary loop. In each generation:
1.  The population is specialized and evaluated.
2.  The best individuals from complementary niches are selected for mating.
3.  A new, evolved "child" model is created using fitness-weighted merging.
4.  **Survival of the Fittest:** The child competes with the previous generation, and only the fittest individuals survive to the next generation (elitism).

The simulation is also robust against loss of diversity; if no suitable mates can be found, it will skip a generation of mating and continue.

## Project Structure

- `main.py`: The main script that orchestrates the evolutionary experiment.
- `evolution.py`: Contains the core logic for the evolutionary process, including the fitness-weighted merging strategy.
- `model.py`: Defines the `CifarCNN` architecture.
- `data.py`: Handles loading and creating niches for the CIFAR-10 dataset.
- `requirements.txt`: The required Python libraries.
- `.gitignore`: Ignores the downloaded dataset and other artifacts.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the experiment:**
    ```bash
    python3 main.py
    ```

You can tweak `POPULATION_SIZE` and `NUM_GENERATIONS` in `main.py`, or change the merging strategy in the call to `merge()` inside the main loop. The default is now `'fitness_weighted'`.
