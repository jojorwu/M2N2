# M2N2: A Simplified Implementation

## 1. Purpose

This project provides a simplified, educational implementation of the concepts from Sakana AI's research paper on **M2N2 (Model Merging of Natural Niches)**. It offers a clear, runnable example of how a population of neural networks can evolve through specialization, intelligent mating, and merging to produce a more capable, generalist model.

This implementation is designed as a learning tool to understand the core principles of the M2N2 paper. It supports multiple model architectures (CNN, ResNet, LLM) and datasets (CIFAR-10, MNIST, Banking77 for LLMs) and features several different model merging and selection strategies to demonstrate the evolutionary process in a transparent and accessible way.

## 2. How It Works: The Evolutionary Algorithm

The simulation follows a generational loop, where each step is designed to mimic principles of natural evolution to improve the overall fitness of the model population.

1.  **Initialization (Niche Adaptation):** The simulation begins by creating a population of specialist neural networks. Each model is trained exclusively on a single class from the chosen dataset (e.g., for CIFAR-10, one model sees only 'airplane' images, another sees only 'cat' images). This forces each model to become an expert in its narrow "niche."

2.  **Evaluation (Measuring Fitness):** In every generation, all models in the population are evaluated against the **full, general test set** for the chosen dataset. The resulting accuracy score represents the model's "fitness." A specialist model will perform well on its own class but poorly on others, while a merged model is expected to have more balanced, general-purpose performance.

3.  **Intelligent Mating (Parent Selection):** The default `HealingMateSelectionStrategy` is used to select complementary parents:
    *   **Parent 1** is chosen as the model with the highest overall fitness.
    *   The algorithm then analyzes Parent 1 to find its "weakest" class.
    *   **Parent 2** is chosen as the specialist model for that weakest class.
    This "healing" strategy ensures that merging is targeted at improving a model's specific weaknesses.

4.  **Crossover (Model Merging):** The two parent models are merged into a new child model using one of several available strategies, configurable in `config.yaml`:
    *   **Average:** Simple weight averaging.
    *   **Fitness-Weighted:** Averages weights, giving more influence to the fitter parent.
    *   **Layer-Wise:** Randomly selects entire layers from either parent.
    *   **Sequential Constructive:** Intelligently builds a child layer by layer, keeping changes only if they improve validation fitness.

5.  **Mutation & Fine-tuning:** To introduce genetic diversity, the child's weights are randomly mutated with a small, decaying probability. It is then fine-tuned on the full, general dataset to help it learn how to integrate the knowledge from its two distinct parents.

6.  **Selection (Survival of the Fittest):** The newly created and fine-tuned child is added to the population pool. The default `ReplaceWorstStrategy` (an elitist strategy) ranks all models by their fitness, and only the top performers survive into the next generation.

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
    ```bash
    pip install -r src/requirements.txt
    ```

### Execution

The experiment is run in two separate terminals:

**Terminal 1: Run the Simulation**
```bash
python3 -m src.main
```

**Terminal 2: Launch the Monitoring Dashboard**
```bash
streamlit run src/dashboard.py
```

### Interactive Controls

The dashboard sidebar allows you to modify the simulation **in real-time**. Changes will take effect at the start of the next generation. You can adjust evolutionary settings, merge strategies, and optimizer parameters.

## 4. Configuration

All experiment parameters are managed in the **`config.yaml`** file. Key parameters include:
- `model_name`: The neural network architecture to use (`CNN`, `RESNET`, `LLM`).
- `dataset_name`: The dataset for the experiment (`CIFAR10`, `MNIST`, `LLM`).
- `num_generations` & `population_size`: Core evolutionary parameters.
- `mate_selection_strategy`, `merge_strategy`, `generation_strategy`: The algorithms to use for key evolutionary steps.
- `mutation_rate`, `initial_mutation_strength`, `mutation_decay_factor`: Control genetic diversity.
- `optimizer_config` & `scheduler_config`: Control the fine-tuning process.
- `subset_percentage`: The fraction of the dataset to use for quick tests.

## 5. Iterative Evolution: Saving and Loading

- **Saving:** At the end of a run, the final population of models is automatically saved as `.pth` files in the `src/pretrained_models/` directory.
- **Loading:** When `main.py` is executed, it first checks the `src/pretrained_models/` directory. If models are found, it loads them as the initial population.

**To start a fresh experiment**, simply delete the contents of the `src/pretrained_models/` directory.

## 6. Developer Notes

### Purpose of `if __name__ == '__main__':`
The `if __name__ == '__main__':` blocks in `data.py`, `model.py`, and `visualization.py` contain example code that demonstrates how to use the functions or classes within that file. This code only runs when the script is executed directly, not when it is imported as a module.

## 7. Project Structure

```
.
├── .gitignore
├── README.md
├── config.yaml
└── src/
    ├── __init__.py
    ├── pretrained_models/
    ├── main.py
    ├── dashboard.py
    ├── simulator.py
    ├── evolution.py
    ├── model.py
    ├── model_factory.py
    ├── model_wrapper.py
    ├── data.py
    ├── enums.py
    ├── utils.py
    ├── visualization.py
    ├── constants.py
    ├── logger_config.py
    ├── merge_strategies.py
    ├── selection_strategies.py
    ├── generation_strategies.py
    └── requirements.txt
```

## 8. License

This project is licensed under the MIT License.
