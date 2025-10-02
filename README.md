# M2N2: A Simplified Implementation (CIFAR-10 Version)

## 1. Purpose

This project provides a simplified, educational implementation of the concepts from Sakana AI's research paper on **M2N2 (Model Merging of Natural Niches)**. It offers a clear, runnable example of how a population of neural networks can evolve through specialization, intelligent mating, and merging to produce a more capable, generalist model.

This implementation is designed as a learning tool to understand the core principles of the M2N2 paper. It uses the CIFAR-10 dataset and a specific, advanced mating strategy to demonstrate the evolutionary process in a transparent and accessible way.

## 2. How It Works: The Evolutionary Algorithm

The simulation follows a generational loop, where each step is designed to mimic principles of natural evolution to improve the overall fitness of the model population.

1.  **Initialization (Niche Adaptation):** The simulation begins by creating a population of 10 specialist Convolutional Neural Networks (CNNs). Each model is trained exclusively on a single class from the CIFAR-10 dataset (e.g., one model sees only 'airplane' images, another sees only 'cat' images). This forces each model to become an expert in its narrow "niche."

2.  **Evaluation (Measuring Fitness):** In every generation, all models in the population are evaluated against the **full, general CIFAR-10 test set**. The resulting accuracy score represents the model's "fitness." A specialist model will perform well on its own class but poorly on others, while a merged model is expected to have more balanced, general-purpose performance.

3.  **Intelligent Mating (Parent Selection):** To create a new "child" model, a sophisticated mating strategy is used to select two parents:
    *   **Parent 1** is chosen as the model with the highest overall fitness in the current population.
    *   The algorithm then analyzes Parent 1 to find its "weakest" class (the class it classifies with the lowest accuracy).
    *   **Parent 2** is chosen as the specialist model for that weakest class.
    This strategy ensures that merging is targeted at improving a model's specific weaknesses, promoting the creation of a more robust child.

4.  **Crossover (Fitness-Weighted Merging):** The two parent models are merged into a new child model. Their weights are combined using a 'fitness_weighted' average, where the parent with the higher fitness score contributes more to the child's final parameters.

5.  **Mutation & Fine-tuning:** To introduce genetic diversity, the child's weights are randomly mutated with a small probability. It is then fine-tuned on the full, general dataset to help it learn how to integrate the knowledge from its two distinct parents into a single, cohesive network.

6.  **Selection (Survival of the Fittest):** The newly created and fine-tuned child is added to the population pool. All models are then ranked by their fitness, and only the top performers (a number equal to the original population size) survive into the next generation. This elitist selection ensures that the population's average fitness tends to increase over time.

## 3. How to Run and Monitor the Experiment

This project uses a decoupled architecture where the simulation runs as a background process and a Streamlit dashboard visualizes the results in real-time.

### Prerequisites
- Python 3.8+
- Pip

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This will install PyTorch, Streamlit, and other necessary packages.
    ```bash
    pip install -r src/requirements.txt
    ```

### Execution

The experiment is run in two separate terminals:

**Terminal 1: Run the Simulation**

This command starts the main evolutionary process. It will run in the background, performing the calculations for each generation and saving the results to a `fitness_log.csv` file.

```bash
python3 -m src.main
```

**Terminal 2: Launch the Monitoring Dashboard**

This command starts the Streamlit web server and opens the dashboard in your browser. The dashboard will automatically read `fitness_log.csv` and update the charts as the simulation progresses.

```bash
streamlit run src/dashboard.py
```

The dashboard provides a real-time view of the best and average fitness per generation.

## 4. Configuration

All experiment parameters are managed in the **`config.yaml`** file. This centralized approach allows you to easily modify the simulation without changing the source code.

Key parameters you can change include:
- `model_config`: The neural network architecture to use (e.g., 'RESNET').
- `dataset_name`: The dataset for the experiment (e.g., 'CIFAR10').
- `num_generations`: The number of evolutionary cycles to run.
- `population_size`: The number of models in the population.
- `merge_strategy`: The algorithm used to merge parent models.
- `subset_percentage`: The fraction of the dataset to use. A value of `0.1` (10%) is used by default for quick tests, while `1.0` uses the full dataset for robust results.
- `specialize_epochs` / `finetune_epochs`: The number of training epochs for different phases of the simulation.

## 5. Iterative Evolution: Saving and Loading

This implementation supports an iterative workflow, allowing you to continue an experiment from where it left off.

- **Saving:** At the end of a run, the final population of models is automatically saved as `.pth` files in the `src/pretrained_models/` directory. Each filename includes the model's niche and its final fitness score.

- **Loading:** When `main.py` is executed, it first checks the `src/pretrained_models/` directory. If models are found, it loads them as the initial population. If the directory is empty, it initializes and trains a new population of specialists from scratch.

**To start a fresh experiment**, simply delete the contents of the `src/pretrained_models/` directory before running the script.

## 6. Developer Notes

### Purpose of `if __name__ == '__main__':`
The `if __name__ == '__main__':` blocks in `data.py` and `model.py` contain example code that demonstrates how to use the functions or classes within that file. This code only runs when the script is executed directly, not when it is imported as a module into another script (like `main.py` or `evolution.py`). This makes the files self-documenting and easy to test independently.

## 7. Project Structure

```
.
├── .gitignore
├── README.md
├── config.yaml               # Central configuration file for the simulation.
├── fitness_log.csv           # Log file generated by the simulation, read by the dashboard.
└── src/
    ├── main.py               # Main script to run the evolutionary experiment.
    ├── dashboard.py          # Streamlit dashboard for monitoring results.
    ├── simulator.py          # Main EvolutionSimulator class.
    ├── evolution.py          # Core logic for selection, merging, and mutation.
    ├── model.py              # Defines the neural network architectures.
    ├── data.py               # Handles data loading and processing.
    ├── enums.py              # Enumerations for models, datasets, etc.
    ├── utils.py              # Utility functions (e.g., seeding).
    ├── model_wrapper.py      # Wrapper class for PyTorch models.
    ├── merge_strategies.py   # Different strategies for model merging.
    └── requirements.txt      # Required Python libraries.
```

## 8. License

This project is licensed under the MIT License. See the details below.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.