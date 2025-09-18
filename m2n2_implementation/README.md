# M2N2: A Simplified Implementation (CIFAR-10 Version)

This project is a simplified, educational implementation of the concepts from Sakana AI's research paper on M2N2 (Model Merging of Natural Niches). It demonstrates how a population of neural networks can be evolved through specialization, intelligent mating, and merging to produce a more capable, generalist model.

This version of the project has been adapted to work with the **CIFAR-10 dataset**.

## Core Concepts Demonstrated

- **Specialization (Resource Competition):** We create a population of CNN models and train them on different "niches" of the CIFAR-10 dataset. The niches are "animals" (bird, cat, deer, etc.) and "vehicles" (airplane, automobile, etc.).
- **Intelligent Mating:** The best-performing model from each niche are selected as "parents".
- **Merging (Crossover):** The parent models are merged into a "child" model by averaging their weights.
- **Mutation & Fine-tuning:** The child is mutated and then fine-tuned on the full dataset to integrate its parents' knowledge.

## Multi-Generational Evolution

This implementation supports a multi-generational evolutionary loop. The `main.py` script will run for a configurable `NUM_GENERATIONS`. In each generation, the following happens:
1.  The existing population is specialized and evaluated.
2.  The best individuals from complementary niches are selected for mating.
3.  A new, evolved "child" model is created.
4.  **Survival of the Fittest:** The child competes with the previous generation. The fittest individuals from this combined pool are selected to form the population for the next generation (a process called elitism).

This process allows the population's overall fitness to improve over time. The simulation is also robust against loss of diversity; if no suitable mates can be found, it will skip a generation of mating rather than crashing.

## Project Structure

- `main.py`: The main script to run the evolutionary experiment.
- `evolution.py`: Contains the core logic for the evolutionary process.
- `model.py`: Defines the `CifarCNN` architecture suitable for CIFAR-10.
- `data.py`: Handles loading and niching the CIFAR-10 dataset.
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

You can tweak the `POPULATION_SIZE` and `NUM_GENERATIONS` constants in `main.py` to run a longer or larger simulation. Note that the current settings are minimal to ensure completion in a constrained environment, as CIFAR-10 is more computationally demanding than MNIST.
