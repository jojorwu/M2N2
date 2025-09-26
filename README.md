# M2N2: A Simplified Implementation (CIFAR-10 Version)

## 1. Purpose

This project provides a simplified, educational implementation of the concepts from Sakana AI's research paper on **M2N2 (Model Merging of Natural Niches)**. It offers a clear, runnable example of how a population of neural networks can evolve through specialization, intelligent mating, and merging to produce a more capable, generalist model.

This implementation is designed as a learning tool to understand the core principles of the M2N2 paper. It uses the CIFAR-10 dataset and a specific, advanced mating strategy to demonstrate the evolutionary process in a transparent and accessible way.

## 2. Core Concepts Demonstrated

The simulation is built around the following key concepts:

- **Specialization (Niche Adaptation):** The initial population consists of 10 specialist Convolutional Neural Networks (CNNs). Each model is trained on only a single class from the CIFAR-10 dataset (e.g., one model for 'airplane', one for 'cat'), forcing it to become an expert in that "niche."

- **Advanced Intelligent Mating:** To create a new "child" model, a sophisticated mating strategy is used:
    1. The best-performing model in the population (based on overall accuracy on the full test set) is selected as **Parent 1**.
    2. This model is analyzed to find its "weakest" class (the one it performs worst on).
    3. The specialist model for that weakest class is chosen as **Parent 2**.
    This ensures that merging is always targeted at improving a model's specific weaknesses.

- **Fitness-Weighted Merging (Crossover):** The parent models are merged into a child using a 'fitness_weighted' average of their parameters. The parent with the higher fitness score has more influence on the child's final weights.

- **Mutation & Fine-tuning:** To introduce genetic diversity, the child's weights are randomly mutated. It is then fine-tuned on the full, general dataset to help it learn to integrate the knowledge from its parents.

- **Elitism (Survival of the Fittest):** The simulation runs for multiple generations. In each generation, the new child competes with the existing population. Only the top performers (based on fitness) survive into the next generation.

## 3. How to Run the Experiment

### Prerequisites
- Python 3.x
- Pip

### Setup and Execution

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    All required libraries are listed in `m2n2_implementation/requirements.txt`.
    ```bash
    pip install -r m2n2_implementation/requirements.txt
    ```

4.  **Run the experiment:**
    Execute the main script from the root directory of the project.
    ```bash
    python3 m2n2_implementation/main.py
    ```
    The script will print its progress for each generation, including specialization, evaluation, mating, and the creation of the next generation.

### Performance Note
To ensure the script runs quickly, it currently operates on a small (10%) subset of the CIFAR-10 data. This can be adjusted by modifying the `subset_percentage` argument in the calls to `get_cifar10_dataloaders` within the `m2n2_implementation/evolution.py` file.

## 4. Iterative Evolution: Loading and Saving Models

This implementation supports an iterative workflow where the evolved population from one run can be used as the starting point for the next.

- **Saving:** At the end of each experiment, the final population of models is automatically saved as `.pth` files in the `m2n2_implementation/pretrained_models/` directory. The filename includes the model's niche and final fitness score.

- **Loading:** When the `main.py` script is run, it first checks the `m2n2_implementation/pretrained_models/` directory.
    - If it finds model files, it will load them as the initial population, allowing you to continue evolution from where you left off.
    - If the directory is empty, it will initialize and train a new population of specialists from scratch.

**To start a fresh experiment**, simply clear the contents of the `m2n2_implementation/pretrained_models/` directory before running the script.

## 5. Project Structure

```
.
├── .gitignore
├── README.md                 # This file
└── m2n2_implementation/
    ├── main.py               # Main script to run the evolutionary experiment.
    ├── evolution.py          # Core logic for selection, merging, and mutation.
    ├── model.py              # Defines the CifarCNN architecture.
    ├── data.py               # Handles loading and subsetting of CIFAR-10.
    ├── requirements.txt      # Required Python libraries.
    └── pretrained_models/    # Directory for saving and loading model populations.
```

## 6. License

This project is licensed under the MIT License. See the details below.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.