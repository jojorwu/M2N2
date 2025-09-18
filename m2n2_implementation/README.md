# M2N2: A Simplified Implementation (CIFAR-10 Version)

This project is a simplified, educational implementation of the concepts from Sakana AI's research paper on M2N2 (Model Merging of Natural Niches). It demonstrates how a population of neural networks can be evolved through specialization, intelligent mating, and merging to produce a more capable, generalist model.

This version of the project runs on the **CIFAR-10 dataset** and uses an advanced intelligent mating strategy.

## Core Concepts Demonstrated

- **Specialization (10 Niches):** The population consists of 10 specialist CNNs, each trained on a single class from the CIFAR-10 dataset (e.g., one model for 'airplane', one for 'cat', etc.).
- **Advanced Intelligent Mating:** A more sophisticated mating strategy is used:
    1. The best-performing model in the population (based on overall accuracy) is selected as Parent 1.
    2. This model is analyzed to find its "weakest" class (the one it performs worst on).
    3. The specialist model for that weakest class is chosen as Parent 2.
    This ensures that merging is always targeted at improving a model's specific weaknesses.
- **Fitness-Weighted Merging (Crossover):** The parent models are merged into a "child" using a 'fitness_weighted' average, where the fitter parent has more influence.
- **Mutation & Fine-tuning:** The child is mutated and then fine-tuned on the full dataset.

## Multi-Generational Evolution

The simulation runs for multiple generations. In each generation, a new child is created via the process above and competes with the existing population for survival into the next generation based on fitness (elitism).

## Project Structure

- `main.py`: The main script that orchestrates the evolutionary experiment.
- `evolution.py`: Contains the core logic, including the advanced intelligent mating strategy.
- `model.py`: Defines the `CifarCNN` architecture.
- `data.py`: Handles loading the CIFAR-10 dataset and can create subsets for faster execution.
- `requirements.txt`: The required Python libraries.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the experiment:**
    ```bash
    python3 main.py
    ```

Note: To prevent timeouts in constrained environments, the script currently runs on a small (10%) subset of the CIFAR-10 data. This can be changed by modifying the `subset_percentage` argument in the calls to `get_cifar10_dataloaders` within the `evolution.py` file.
