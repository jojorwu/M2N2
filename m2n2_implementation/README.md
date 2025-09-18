# M2N2: A Simplified Implementation

This project is a simplified, educational implementation of the concepts from Sakana AI's research paper on M2N2 (Model Merging of Natural Niches). It demonstrates how a population of neural networks can be evolved through specialization, intelligent mating, and merging to produce a more capable, generalist model.

## Core Concepts Demonstrated

- **Specialization (Resource Competition):** We create a population of simple CNN models and train them on different "niches" of the MNIST dataset (e.g., one group on digits 0-4, another on 5-9). This forces them to become specialists.
- **Intelligent Mating:** After evaluating the specialists, we select the best-performing model from each niche to act as "parents". This is a simplified form of selecting for complementary strengths.
- **Merging (Crossover):** The two parent models are merged into a single "child" model by averaging their weights.
- **Mutation:** Small random noise is added to the child's weights to ensure genetic diversity.
- **Fine-tuning:** The newly created child is fine-tuned on the entire dataset, which allows it to learn how to properly integrate the specialized knowledge from its parents.

## Project Structure

- `main.py`: The main script to run the entire evolutionary experiment. It initializes the population, runs the specialization, evolution, and evaluation steps, and prints a final summary.
- `evolution.py`: Contains the core logic for the evolutionary process, including functions for specializing, evaluating, selecting, merging, mutating, and fine-tuning models.
- `model.py`: Defines the `SimpleCNN` architecture used for all models in the population.
- `data.py`: Contains the `get_mnist_dataloaders` function, which handles loading the MNIST dataset and splitting it into niches.
- `requirements.txt`: The required Python libraries to run this project.
- `.gitignore`: A gitignore file to exclude the downloaded dataset and other artifacts from version control.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the experiment:**
    ```bash
    python3 main.py
    ```

The script will output the results of the experiment, showing the fitness of the specialist parents and the final, superior fitness of the evolved child model.
