"""Implements the core evolutionary algorithm for the M2N2 simulation.

This module contains the logic for the main steps of the evolutionary
process, including model specialization, evaluation, mate selection,
merging (crossover), mutation, and generational selection. It operates on
`ModelWrapper` objects, which encapsulate the neural network models and their
associated metadata, conforming to Google's Python docstring style.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
from .model import CifarCNN, MnistCNN, LLMClassifier, ResNetClassifier
from .data import get_dataloaders
from .merge_strategies import (
    AverageMergeStrategy,
    FitnessWeightedMergeStrategy,
    LayerWiseMergeStrategy,
    SequentialConstructiveMergeStrategy,
)
from .model_wrapper import ModelWrapper
from .utils import _calculate_accuracy
from typing import List, Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader
from torch import nn

logger = logging.getLogger("M2N2_SIMULATOR")
import copy
import random
from tqdm import tqdm

def _run_training_epoch(model_wrapper: ModelWrapper, optimizer: optim.Optimizer, train_loader: DataLoader, scaler: torch.cuda.amp.GradScaler, precision: str, description: str) -> float:
    """Runs a single training epoch for a given model and returns the average loss."""
    model_wrapper.model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc=description)
    for batch in pbar:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(precision == '16' and 'cuda' in model_wrapper.device)):
            if model_wrapper.model_name == 'LLM':
                input_ids = batch['input_ids'].to(model_wrapper.device)
                attention_mask = batch['attention_mask'].to(model_wrapper.device)
                labels = batch['labels'].to(model_wrapper.device)
                outputs = model_wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs, labels)
            else:
                data, target = batch
                data = data.to(model_wrapper.device)
                target = target.to(model_wrapper.device)
                if precision == '64':
                    data = data.double()
                output = model_wrapper.model(data)
                loss = F.cross_entropy(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()
        pbar.set_postfix({'train_loss': f"{loss.item():.4f}"})

    return total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

def specialize(model_wrapper: ModelWrapper, dataset_name: str, epochs: int = 1, precision: str = '32', seed: Optional[int] = None, learning_rate: float = 0.001, subset_percentage: float = 0.1) -> None:
    """Trains a model in-place on its specialized data niche.

    This simulates the "resource competition" phase where a model becomes an
    expert in a specific area. It supports different training precisions.

    Args:
        model_wrapper (ModelWrapper): The model to be trained in-place.
        dataset_name (str): The name of the dataset to use for training.
        epochs (int, optional): The number of training epochs. Defaults to 1.
        precision (str, optional): The training precision ('16', '32', '64').
            Defaults to '32'.
        seed (int, optional): A seed for the random number generator to
            ensure deterministic data splitting. Defaults to None.
        learning_rate (float, optional): The learning rate for the optimizer.
            Defaults to 0.001.
        subset_percentage (float, optional): The fraction of the training data
            to use. Defaults to 0.1.
    """
    logger.info(f"Specializing model on niche {model_wrapper.niche_classes} for {epochs} epoch(s) with {precision}-bit precision...")

    train_loader, _, _ = get_dataloaders(
        dataset_name=dataset_name,
        model_name=model_wrapper.model_name,
        niche_classes=model_wrapper.niche_classes,
        subset_percentage=subset_percentage,
        seed=seed
    )
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=learning_rate)

    # Handle precision
    if precision == '64':
        model_wrapper.model.double()

    scaler = torch.cuda.amp.GradScaler(enabled=(precision == '16' and 'cuda' in model_wrapper.device))

    for epoch in range(epochs):
        logger.info(f"  - Epoch {epoch + 1}/{epochs}")
        _run_training_epoch(
            model_wrapper,
            optimizer,
            train_loader,
            scaler,
            precision,
            f"Specializing Niche {model_wrapper.niche_classes}"
        )

    # Mark fitness as not current, as the model has been modified.
    model_wrapper.fitness_is_current = False
    logger.info("Specialization complete.")


def _get_fitness_score(model_wrapper: ModelWrapper, dataset_name: str, subset_percentage: float = 1.0, seed: Optional[int] = None) -> float:
    """Calculates and returns the fitness score for a model on the test set.

    This is a lightweight, side-effect-free version of the `evaluate`
    function. It calculates the accuracy on the full test set but does *not*
    update the `fitness` attribute of the model wrapper or print any
    output. This makes it suitable for repeated internal use.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.
        dataset_name (str): The name of the dataset to use for evaluation.
        subset_percentage (float, optional): The fraction of the test set to use for evaluation. Defaults to 1.0.
        seed (int, optional): A seed for the random number generator to
            ensure deterministic data splitting. Defaults to None.

    Returns:
        float: The calculated accuracy (fitness) of the model as a percentage.
    """
    # We always evaluate on the full test set to measure general performance
    _, _, test_loader = get_dataloaders(dataset_name=dataset_name, model_name=model_wrapper.model_name, subset_percentage=subset_percentage, validation_split=0, seed=seed) # No validation split needed here
    return _calculate_accuracy(model_wrapper, test_loader)

def evaluate(model_wrapper: ModelWrapper, dataset_name: str, subset_percentage: float = 1.0, seed: Optional[int] = None) -> float:
    """Evaluates fitness on the full test set and updates the wrapper.

    This function skips evaluation if the model's fitness is already
    marked as current. Otherwise, it calculates the accuracy on the test
    set and updates the `fitness` and `fitness_is_current` attributes.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.
        dataset_name (str): The name of the dataset to use for evaluation.
        subset_percentage (float, optional): The fraction of the test set to use for evaluation. Defaults to 1.0.
        seed (int, optional): A seed for the random number generator to
            ensure deterministic data splitting. Defaults to None.

    Returns:
        float: The calculated accuracy (fitness) of the model as a percentage.
    """
    if model_wrapper.fitness_is_current:
        logger.debug(f"  - Skipping evaluation for model with up-to-date fitness: {model_wrapper.fitness:.2f}%")
        return model_wrapper.fitness

    accuracy = _get_fitness_score(model_wrapper, dataset_name=dataset_name, subset_percentage=subset_percentage, seed=seed)
    model_wrapper.fitness = accuracy
    model_wrapper.fitness_is_current = True
    return accuracy

def evaluate_by_class(model_wrapper: ModelWrapper, dataset_name: str, subset_percentage: float = 1.0, seed: Optional[int] = None) -> List[float]:
    """Evaluates a model's accuracy on each individual class.

    This function is used to identify a model's strengths and weaknesses,
    which is crucial for the advanced mate selection strategy. It does not
    modify the model wrapper.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.
        dataset_name (str): The name of the dataset to use for evaluation.
        subset_percentage (float, optional): The fraction of the test set to use for evaluation. Defaults to 1.0.
        seed (int, optional): A seed for the random number generator to
            ensure deterministic data splitting. Defaults to None.

    Returns:
        list[float]: A list of accuracy percentages, where the index of the
            list corresponds to the class index.
    """
    # We always evaluate on the full test set to measure general performance
    _, _, test_loader = get_dataloaders(dataset_name=dataset_name, model_name=model_wrapper.model_name, subset_percentage=subset_percentage, validation_split=0, seed=seed) # No validation split needed here
    model_wrapper.model.eval()

    num_classes = model_wrapper.model.num_classes
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        for batch in test_loader:
            if model_wrapper.model_name == 'LLM':
                input_ids = batch['input_ids'].to(model_wrapper.device)
                attention_mask = batch['attention_mask'].to(model_wrapper.device)
                target = batch['labels'].to(model_wrapper.device)
                output = model_wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                data, target = batch
                data, target = data.to(model_wrapper.device), target.to(model_wrapper.device)
                output = model_wrapper.model(data)

            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)

    return class_accuracies

def select_mates(population: List[ModelWrapper], dataset_name: str, subset_percentage: float = 1.0, seed: Optional[int] = None) -> Tuple[Optional[ModelWrapper], Optional[ModelWrapper]]:
    """Selects a complementary pair of parents using an advanced strategy.

    This function promotes "healing" by pairing a strong model with a model
    that is an expert in the first model's weakest area. The strategy is:
    1.  Parent 1 is chosen as the model with the highest overall fitness.
    2.  Parent 1's performance is analyzed to find its weakest class.
    3.  Parent 2 is chosen as the specialist model for that weakest class.
    4.  A fallback is used if a suitable specialist is not found.

    This function prints its selection logic to the console.

    Args:
        population (list[ModelWrapper]): The current population of models.
        dataset_name (str): The name of the dataset to use for evaluation.
        subset_percentage (float, optional): The fraction of the test set to use for evaluation. Defaults to 1.0.
        seed (int, optional): A seed for the random number generator to
            ensure deterministic data splitting. Defaults to None.

    Returns:
        tuple[ModelWrapper | None, ModelWrapper | None]: A tuple containing
            the two selected parents. If a suitable pair cannot be found
            (e.g., population is too small), elements can be `None`.
    """
    logger.info("Selecting mates with advanced strategy...")
    if not population:
        return None, None

    # 1. Find the best overall model in the population to be Parent 1.
    parent1 = max(population, key=lambda m: m.fitness)
    logger.info(f"  - Parent 1 is the population's best model (Fitness: {parent1.fitness:.2f}%)")

    # 2. Analyze Parent 1 to find its weakest class.
    logger.info("  - Analyzing Parent 1's performance by class...")
    class_accuracies = evaluate_by_class(parent1, dataset_name=dataset_name, subset_percentage=subset_percentage, seed=seed)
    min_accuracy = min(class_accuracies)
    weakest_indices = [i for i, acc in enumerate(class_accuracies) if acc == min_accuracy]
    weakest_class_index = random.choice(weakest_indices)
    logger.info(f"  - Parent 1's weakest class is {weakest_class_index} (Accuracy: {class_accuracies[weakest_class_index]:.2f}%)")

    # 3. Find the specialist for that weakest class to be Parent 2.
    parent2 = None
    # Ensure Parent 2 is not the same model as Parent 1.
    specialist_candidates = [
        m for m in population if m.niche_classes == [weakest_class_index] and m is not parent1
    ]

    if specialist_candidates:
        # From the candidates, pick the one with the highest fitness.
        parent2 = max(specialist_candidates, key=lambda m: m.fitness)
        logger.info(f"  - Found specialist for class {weakest_class_index} as Parent 2 (Fitness: {parent2.fitness:.2f}%)")
    else:
        # Fallback: if no suitable specialist is found, pick the second-best model overall,
        # ensuring it's not the same instance as Parent 1.
        logger.info("  - No suitable specialist found. Using second-best model as fallback Parent 2.")
        sorted_population = sorted(population, key=lambda m: m.fitness, reverse=True)

        # Find the first model in the sorted list that is not Parent 1.
        parent2 = next((model for model in sorted_population if model is not parent1), None)

        if parent2 is None:
            # This happens if all models in the population are the same instance
            # or if there's only one model.
            logger.info("  - Not enough distinct models in population to select a second parent.")
            return parent1, None

    if parent1 and parent2:
        return parent1, parent2
    else:
        # This case should be rare given the fallbacks, but is here for safety.
        logger.info("  - Could not select a pair of parents.")
        return None, None

def merge(parent1: ModelWrapper, parent2: ModelWrapper, strategy: str = 'average', validation_loader: Optional[DataLoader] = None, seed: Optional[int] = None, dampening_factor: float = 25.0) -> ModelWrapper:
    """
    Merges two parent models into a new child model using a specified strategy.
    """
    logger.info(f"Merging parent models to create child using '{strategy}' strategy...")

    strategy_map: Dict[str, Any] = {
        'average': AverageMergeStrategy,
        'fitness_weighted': FitnessWeightedMergeStrategy,
        'layer-wise': LayerWiseMergeStrategy,
        'sequential_constructive': SequentialConstructiveMergeStrategy,
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    # Prepare arguments for the strategy constructor
    strategy_args = {}
    if strategy == 'fitness_weighted':
        strategy_args['dampening_factor'] = dampening_factor
    elif strategy == 'layer-wise':
        strategy_args['seed'] = seed

    # Instantiate the strategy and merge
    merge_strategy = strategy_map[strategy](**strategy_args)
    child_model_state_dict = merge_strategy.merge(parent1, parent2, validation_loader)

    # Create and return the new child model
    num_classes = parent1.model.num_classes
    child_wrapper = ModelWrapper(model_name=parent1.model_name, niche_classes=list(range(num_classes)), device=parent1.device)
    child_wrapper.model.load_state_dict(child_model_state_dict)
    logger.info("Merging complete.")
    return child_wrapper

def mutate(model_wrapper: ModelWrapper, generation: int, mutation_rate: float = 0.01, initial_mutation_strength: float = 0.1, decay_factor: float = 0.9) -> ModelWrapper:
    """Applies random, adaptively scaled Gaussian mutations to a model's weights.

    This function introduces genetic diversity by altering a fraction of the
    model's weights. The mutation strength is adaptive, decaying
    exponentially with each generation. This allows for larger exploratory
    changes in early generations and smaller, more precise changes later on.
    The mutation is applied in-place.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to mutate.
        generation (int): The current generation number, used to calculate
            the decaying mutation strength.
        mutation_rate (float, optional): The probability that any given
            weight will be chosen for mutation. Defaults to 0.01.
        initial_mutation_strength (float, optional): The initial standard
            deviation for the mutation noise. Defaults to 0.1.
        decay_factor (float, optional): The factor by which the mutation
            strength decays each generation (e.g., 0.9 means 10% decay).
            Defaults to 0.9.

    Returns:
        ModelWrapper: The same model wrapper that was passed in, allowing
            for method chaining.
    """
    # Calculate the decayed mutation strength for the current generation
    decayed_strength = initial_mutation_strength * (decay_factor ** generation)
    logger.info(f"Mutating child model (Gen: {generation}, Strength: {decayed_strength:.4f})...")

    with torch.no_grad():
        for param in model_wrapper.model.parameters():
            if len(param.shape) > 1: # Mutate only multi-dimensional layers (conv, linear)
                # Create a random mask to decide which weights to mutate
                mutation_mask = (torch.rand(param.shape) < mutation_rate).to(model_wrapper.device)
                # Generate random noise scaled by the decayed strength
                mutation = torch.randn(param.shape).to(model_wrapper.device) * decayed_strength
                # Apply the mutation where the mask is True
                param.data += mutation * mutation_mask
    # Mark fitness as not current, as the model has been modified.
    model_wrapper.fitness_is_current = False
    logger.info("Mutation complete.")
    return model_wrapper

def create_next_generation(current_population: List[ModelWrapper], new_child: ModelWrapper, population_size: int, dataset_name: str, seed: Optional[int] = None) -> List[ModelWrapper]:
    """Creates the next generation's population using elitist selection.

    This function implements the selection step of the algorithm. It combines
    the existing population with the new child, evaluates the child's
    fitness, and then selects the top individuals to form the next
    generation's population. This function prints its progress to the console.

    Args:
        current_population (list[ModelWrapper]): The list of models in the
            current generation.
        new_child (ModelWrapper): The newly created child model to be
            evaluated and included in the selection pool.
        population_size (int): The maximum size of the population.
        dataset_name (str): The name of the dataset to use for evaluation.
        seed (int, optional): A seed for the random number generator to
            ensure deterministic data splitting. Defaults to None.

    Returns:
        list[ModelWrapper]: A new list of models for the next generation,
            sorted by fitness in descending order.
    """
    logger.info("Creating the next generation...")
    # Evaluate the new child to make sure its fitness is calculated
    evaluate(new_child, dataset_name=dataset_name, seed=seed)

    # Combine the old population with the new child
    full_pool = current_population + [new_child]

    # Sort the entire pool by fitness in descending order
    full_pool.sort(key=lambda x: x.fitness, reverse=True)

    # The next generation consists of the top 'population_size' individuals
    next_generation = full_pool[:population_size]

    logger.info(f"Selected {len(next_generation)} fittest individuals for the next generation.")

    return next_generation

def _calculate_loss(model_wrapper: ModelWrapper, data_loader: DataLoader) -> float:
    """A generic helper to calculate loss on a given data loader."""
    model_wrapper.model.eval()
    total_loss = 0.0
    device = model_wrapper.device

    with torch.no_grad():
        for batch in data_loader:
            if model_wrapper.model_name == 'LLM':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model_wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs, labels)
            else:
                data, target = batch
                data, target = data.to(device), target.to(device)
                if next(model_wrapper.model.parameters()).dtype == torch.float64:
                    data = data.double()
                output = model_wrapper.model(data)
                loss = F.cross_entropy(output, target)
            total_loss += loss.item()

    if len(data_loader) == 0:
        return 0.0
    return total_loss / len(data_loader)


def finetune(model_wrapper: ModelWrapper, dataset_name: str, validation_loader: DataLoader, epochs: int = 3, precision: str = '32', seed: Optional[int] = None, learning_rate: float = 0.001, scheduler_patience: int = 2, scheduler_factor: float = 0.5, subset_percentage: float = 0.1) -> None:
    """Fine-tunes a model in-place on the full dataset with a scheduler.

    This step is crucial for a newly merged child model. It uses an Adam
    optimizer and a `ReduceLROnPlateau` learning rate scheduler, and supports
    different training precisions.

    Args:
        model_wrapper (ModelWrapper): The model to be fine-tuned in-place.
        dataset_name (str): The name of the dataset to use for training.
        validation_loader (DataLoader): A DataLoader for the validation set,
            used to control the learning rate scheduler.
        epochs (int, optional): The number of fine-tuning epochs.
            Defaults to 3.
        precision (str, optional): The training precision ('16', '32', '64').
            Defaults to '32'.
        seed (int, optional): A seed for the random number generator to
            ensure deterministic data splitting. Defaults to None.
        learning_rate (float, optional): The learning rate for the optimizer.
            Defaults to 0.001.
        scheduler_patience (int, optional): The patience for the learning rate
            scheduler. Defaults to 2.
        scheduler_factor (float, optional): The factor for the learning rate
            scheduler. Defaults to 0.5.
        subset_percentage (float, optional): The fraction of the training data
            to use. Defaults to 0.1.
    """
    logger.info(f"Fine-tuning model for {epochs} epoch(s) with {precision}-bit precision and ReduceLROnPlateau scheduler...")

    # We get a train_loader with the full training data (no validation split here)
    train_loader, _, _ = get_dataloaders(
        dataset_name=dataset_name,
        model_name=model_wrapper.model_name,
        subset_percentage=subset_percentage,
        seed=seed,
        validation_split=0.0
    )
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=scheduler_factor)

    if precision == '64':
        model_wrapper.model.double()

    scaler = torch.cuda.amp.GradScaler(enabled=(precision == '16' and 'cuda' in model_wrapper.device))

    for epoch in range(epochs):
        logger.info(f"  - Epoch {epoch + 1}/{epochs}")
        avg_train_loss = _run_training_epoch(
            model_wrapper,
            optimizer,
            train_loader,
            scaler,
            precision,
            "Fine-tuning Child"
        )

        # Calculate validation loss for the scheduler
        avg_val_loss = _calculate_loss(model_wrapper, validation_loader)
        scheduler.step(avg_val_loss)

        logger.info(f"  - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

    # Mark fitness as not current, as the model has been modified.
    model_wrapper.fitness_is_current = False
    logger.info("Fine-tuning complete.")