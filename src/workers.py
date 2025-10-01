"""
Defines worker functions for parallel processing in the M2N2 simulation.

Each function is designed to be a self-contained unit of work that can be
executed in a separate process, avoiding the need to pickle and transfer
large, complex objects like PyTorch models.
"""
from .evolution import ModelWrapper, specialize, _get_fitness_score

def specialize_worker(args):
    """
    A worker function that initializes a model, specializes it on a niche,
    and returns its updated state dictionary.

    Args:
        args (tuple): A tuple containing all the necessary parameters:
            - model_name (str): The name of the model architecture.
            - niche_classes (list[int]): The niche to specialize in.
            - num_classes (int): The number of output classes for the model.
            - device (str): The device to run on ('cpu' or 'cuda').
            - dataset_name (str): The name of the dataset for specialization.
            - epochs (int): The number of epochs for specialization.
            - precision (str): The training precision.
            - seed (int): The random seed for reproducibility.

    Returns:
        dict: The state dictionary of the specialized model's weights.
    """
    model_name, niche_classes, num_classes, device, dataset_name, epochs, precision, seed = args

    # 1. Create a new ModelWrapper instance within the worker process.
    model_wrapper = ModelWrapper(model_name, niche_classes, num_classes, device)

    # 2. Run the specialization process.
    specialized_wrapper = specialize(model_wrapper, dataset_name, epochs, precision, seed)

    # 3. Return the lightweight, picklable state dictionary.
    return specialized_wrapper.model.state_dict()

def evaluate_worker(args):
    """
    A worker function that initializes a model, loads its state, evaluates
    its fitness, and returns the score.

    Args:
        args (tuple): A tuple containing all the necessary parameters:
            - state_dict (dict): The state dictionary of the model to evaluate.
            - model_name (str): The name of the model architecture.
            - niche_classes (list[int]): The model's niche.
            - num_classes (int): The number of output classes for the model.
            - device (str): The device to run on ('cpu' or 'cuda').
            - dataset_name (str): The name of the dataset for evaluation.
            - seed (int): The random seed for reproducibility.

    Returns:
        float: The calculated fitness score of the model.
    """
    state_dict, model_name, niche_classes, num_classes, device, dataset_name, seed = args

    # 1. Create a new ModelWrapper instance.
    model_wrapper = ModelWrapper(model_name, niche_classes, num_classes, device)

    # 2. Load the model's state from the provided state dictionary.
    model_wrapper.model.load_state_dict(state_dict)

    # 3. Calculate and return the fitness score.
    fitness = _get_fitness_score(model_wrapper, dataset_name, seed)
    return fitness