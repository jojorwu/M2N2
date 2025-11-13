"""Implements the core evolutionary algorithm for the M2N2 simulation.

This module contains the logic for the main steps of the evolutionary
process, including model specialization, evaluation, mate selection,
merging (crossover), mutation, and generational selection. It operates on
`ModelWrapper` objects, which encapsulate the neural network models and their
associated metadata.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
from .data import get_dataloaders
from .model_factory import create_model
from .merge_strategies import MergeStrategy
from .model_wrapper import ModelWrapper
from typing import List, Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

# Forward-declare types for type hinting
if "ConfigManager" not in globals():
    from typing import TypeVar
    ConfigManager = TypeVar("ConfigManager")
if "MateSelectionStrategy" not in globals():
    from typing import TypeVar
    MateSelectionStrategy = TypeVar("MateSelectionStrategy")
if "GenerationStrategy" not in globals():
    from typing import TypeVar
    GenerationStrategy = TypeVar("GenerationStrategy")


logger = logging.getLogger("M2N2_SIMULATOR")

def _run_training_epoch(
    model_wrapper: ModelWrapper,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    scaler: torch.cuda.amp.GradScaler,
    precision: str,
    description: str,
    show_progress_bar: bool = True
) -> float:
    """
    Runs a single training epoch for a given model.

    This function iterates through the training data loader, performs the forward
    and backward passes, updates the model weights, and calculates the average
    training loss for the epoch. It supports mixed-precision training via a
    `GradScaler`.

    Args:
        model_wrapper (ModelWrapper): The wrapper for the model to be trained.
        optimizer (optim.Optimizer): The optimizer for updating weights.
        train_loader (DataLoader): The data loader for the training set.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed-precision.
        precision (str): The training precision ('16', '32', or '64').
        description (str): A description for the progress bar.
        show_progress_bar (bool, optional): Whether to display a TQDM progress
            bar. Defaults to True.

    Returns:
        float: The average training loss for the epoch.
    """
    model_wrapper.model.train()
    total_train_loss = 0.0
    use_amp = precision == '16' and 'cuda' in model_wrapper.device

    data_iterator = tqdm(train_loader, desc=description) if show_progress_bar else train_loader

    for batch in data_iterator:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            output, target = model_wrapper._process_batch(batch)
            loss = F.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()
        if show_progress_bar:
            data_iterator.set_postfix({'train_loss': f"{loss.item():.4f}"})

    return total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0


def _run_training_session(
    model_wrapper: ModelWrapper,
    train_loader: DataLoader,
    config_manager: "ConfigManager",
    epochs: int,
    description: str,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None,
    validation_loader: Optional[DataLoader] = None
) -> None:
    """
    Manages a complete training session over multiple epochs.
    This function orchestrates the training process, including running the
    training epochs and, if a scheduler and validation loader are provided,
    stepping the learning rate scheduler based on validation loss.
    Args:
        model_wrapper (ModelWrapper): The model wrapper to be trained.
        train_loader (DataLoader): The data loader for the training data.
        config_manager (ConfigManager): The simulation's configuration manager.
        epochs (int): The number of epochs to train for.
        description (str): A description for the progress bar for each epoch.
        optimizer (optim.Optimizer): The optimizer to use for training.
        scheduler (Optional[optim.lr_scheduler.ReduceLROnPlateau], optional):
            An optional learning rate scheduler. Defaults to None.
        validation_loader (Optional[DataLoader], optional): The data loader for
            the validation set, required if a scheduler is used. Defaults to None.
    """
    if config_manager.precision_config == '64':
        model_wrapper.model.double()

    scaler = torch.cuda.amp.GradScaler(enabled=(config_manager.precision_config == '16' and 'cuda' in model_wrapper.device))

    for epoch in range(epochs):
        logger.info(f"  - Epoch {epoch + 1}/{epochs}")
        avg_train_loss = _run_training_epoch(
            model_wrapper, optimizer, train_loader, scaler,
            config_manager.precision_config, description,
            show_progress_bar=config_manager.show_progress_bar
        )
        if scheduler and validation_loader and len(validation_loader) > 0:
            avg_val_loss = _calculate_loss(model_wrapper, validation_loader)
            scheduler.step(avg_val_loss)
            logger.info(f"  - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

def _setup_and_run_training(
    mode: str,
    model_wrapper: ModelWrapper,
    config_manager: "ConfigManager",
    validation_loader: Optional[DataLoader] = None
) -> None:
    """
    Sets up and runs a training session for specialization or fine-tuning.

    This helper encapsulates the logic for preparing a training run. It
    configures epochs, data loaders, optimizer, and scheduler based on the
    training `mode`, then executes the training.

    Args:
        mode (str): The training mode, either 'specialize' or 'finetune'.
        model_wrapper (ModelWrapper): The model to be trained.
        config_manager (ConfigManager): The configuration manager.
        validation_loader (Optional[DataLoader], optional): A validation data
            loader, required for 'finetune' mode. Defaults to None.

    Raises:
        ValueError: If `mode` is invalid or 'finetune' is missing `validation_loader`.
    """
    if mode == 'specialize':
        epochs = config_manager.specialize_epochs
        description = f"Specializing Niche {model_wrapper.niche_classes}"
        train_loader, _, _, _ = get_dataloaders(
            dataset_name=config_manager.dataset_name, model_name=model_wrapper.model_name,
            niche_classes=model_wrapper.niche_classes,
            subset_percentage=config_manager.subset_percentage,
            seed=config_manager.seed, validation_split=0.0
        )
    elif mode == 'finetune':
        epochs = config_manager.finetune_epochs
        description = "Fine-tuning Child"
        if not validation_loader:
            raise ValueError("Validation loader is required for fine-tuning.")
        train_loader, _, _, _ = get_dataloaders(
            dataset_name=config_manager.dataset_name, model_name=model_wrapper.model_name,
            niche_classes=None, subset_percentage=config_manager.subset_percentage,
            seed=config_manager.seed, validation_split=0.0
        )
    else:
        raise ValueError(f"Invalid mode for training session: {mode}")

    if len(train_loader) == 0:
        logger.warning(f"Skipping {mode} for niche {model_wrapper.niche_classes} as data loader is empty.")
        return

    optimizer = _create_optimizer(model_wrapper, config_manager)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=config_manager.scheduler_patience,
        factor=config_manager.scheduler_factor
    ) if mode == 'finetune' else None

    _run_training_session(
        model_wrapper, train_loader, config_manager, epochs, description,
        optimizer, scheduler, validation_loader
    )

def _create_optimizer(model_wrapper: ModelWrapper, config_manager: "ConfigManager") -> optim.Optimizer:
    """Creates an optimizer based on the provided configuration."""
    return optim.Adam(model_wrapper.model.parameters(), lr=config_manager.learning_rate)

def specialize(model_wrapper: ModelWrapper, config_manager: "ConfigManager") -> None:
    """
    Trains a model in-place on its specialized data niche.

    This function orchestrates the "specialization" phase. It configures a
    training session where a model is trained only on its "niche" subset of
    the data, forcing it to become an expert for that class.

    Args:
        model_wrapper (ModelWrapper): The model to be specialized. Its
            `niche_classes` attribute determines the data subset.
        config_manager (ConfigManager): Provides parameters like epochs.
    """
    logger.info(f"Specializing model on niche {model_wrapper.niche_classes} for {config_manager.specialize_epochs} epoch(s)...")
    _setup_and_run_training('specialize', model_wrapper, config_manager)
    model_wrapper.fitness_is_current = False
    logger.info("Specialization complete.")

def select_mates(
    population: List[ModelWrapper],
    num_pairs: int,
    strategy: "MateSelectionStrategy",
    config_manager: "ConfigManager"
) -> List[Tuple[ModelWrapper, ModelWrapper]]:
    """
    Selects multiple pairs of parents from the population for breeding.
    Delegates the selection logic to a `MateSelectionStrategy` object,
    allowing for different algorithms (e.g., healing, random) to be used.
    Args:
        population (List[ModelWrapper]): The current population of models.
        num_pairs (int): The number of parent pairs to select.
        strategy (MateSelectionStrategy): The strategy object that implements
            the selection logic.
        config_manager (ConfigManager): The simulation's configuration.
    Returns:
        List[Tuple[ModelWrapper, ModelWrapper]]: A list of tuples, where each
            tuple contains two parent models selected for mating.
    """
    return strategy.select_parent_pairs(population, num_pairs, config_manager)

def merge(
    parent1: ModelWrapper,
    parent2: ModelWrapper,
    strategy: "MergeStrategy",
    validation_loader: Optional[DataLoader] = None
) -> ModelWrapper:
    """
    Merges two parent models into a new child model.
    This function orchestrates the "crossover" step. It combines the weights
    of two parents to create a child, with the specific algorithm determined
    by the provided `MergeStrategy`.
    Args:
        parent1 (ModelWrapper): The first parent model.
        parent2 (ModelWrapper): The second parent model.
        strategy (MergeStrategy): The object that implements the merging algorithm.
        validation_loader (Optional[DataLoader], optional): Data loader for
            validation, required by some advanced strategies. Defaults to None.
    Returns:
        ModelWrapper: A new `ModelWrapper` for the created child model.
    """
    logger.info(f"Merging parents using '{strategy.__class__.__name__}'...")
    child_model_state_dict = strategy.merge(parent1, parent2, validation_loader)
    child_model = create_model(parent1.model_name, parent1.model.num_classes, parent1.device)
    child_model.load_state_dict(child_model_state_dict)
    child_wrapper = ModelWrapper(
        model_name=parent1.model_name, model=child_model,
        niche_classes=list(range(parent1.model.num_classes)), device=parent1.device
    )
    logger.info("Merging complete.")
    return child_wrapper

def mutate(
    model_wrapper: ModelWrapper,
    generation: int,
    config_manager: "ConfigManager",
    seed: Optional[int] = None
) -> ModelWrapper:
    """
    Applies deterministic, adaptively scaled Gaussian mutations to a model.
    This function introduces genetic diversity by altering a fraction of the
    model's weights. The process is deterministic if a seed is provided. The
    mutation strength decays over generations.
    Args:
        model_wrapper (ModelWrapper): The model to mutate.
        generation (int): The current generation number, for adaptive strength.
        config_manager (ConfigManager): The configuration manager.
        seed (Optional[int], optional): A seed for the random number
            generator to ensure reproducibility. Defaults to None.
    Returns:
        The mutated model wrapper.
    """
    decayed_strength = config_manager.initial_mutation_strength * (config_manager.mutation_decay_factor ** generation)
    logger.info(f"Mutating child (Gen: {generation}, Strength: {decayed_strength:.4f})...")
    generator = torch.Generator(device=model_wrapper.device)
    if seed is not None:
        generator.manual_seed(seed)
    with torch.no_grad():
        for param in model_wrapper.model.parameters():
            if param.dim() > 1:
                num_to_mutate = int(param.numel() * config_manager.mutation_rate)
                if num_to_mutate > 0:
                    indices = torch.randperm(param.numel(), device=model_wrapper.device, generator=generator)[:num_to_mutate]
                    mutation = torch.randn(num_to_mutate, device=model_wrapper.device, generator=generator) * decayed_strength
                    param.view(-1)[indices] += mutation
    model_wrapper.fitness_is_current = False
    logger.info("Mutation complete.")
    return model_wrapper

def create_next_generation(
    current_population: List[ModelWrapper],
    offspring_pool: List[ModelWrapper],
    strategy: "GenerationStrategy",
    config_manager: "ConfigManager"
) -> List[ModelWrapper]:
    """
    Creates the next generation's population from the current one and offspring.
    Applies a survival strategy (e.g., elitism) to determine which models from
    the combined pool will survive to form the next generation.
    Args:
        current_population (List[ModelWrapper]): Models from the current generation.
        offspring_pool (List[ModelWrapper]): Newly created and trained offspring.
        strategy (GenerationStrategy): The object that implements the survival logic.
        config_manager (ConfigManager): The simulation's configuration.
    Returns:
        List[ModelWrapper]: The list of models for the next generation.
    """
    return strategy.create_next_generation(current_population, offspring_pool, config_manager)

def _calculate_loss(model_wrapper: ModelWrapper, data_loader: DataLoader) -> float:
    """
    Calculates the average loss for a model on a given dataset.
    Runs a model in evaluation mode over all batches in a data loader and
    computes the average cross-entropy loss, typically for validation.
    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.
        data_loader (DataLoader): The data loader with the evaluation dataset.
    Returns:
        float: The average loss. Returns 0.0 if the data loader is empty.
    """
    model_wrapper.model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            output, target = model_wrapper._process_batch(batch)
            total_loss += F.cross_entropy(output, target).item()
    return total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

def finetune(
    model_wrapper: ModelWrapper,
    validation_loader: DataLoader,
    config_manager: "ConfigManager"
) -> None:
    """
    Fine-tunes a model in-place on the full dataset with a learning rate scheduler.

    This function orchestrates the "fine-tuning" phase, typically run after
    merging. The model is trained on the entire dataset to integrate knowledge
    from its parents, with a scheduler adapting the learning rate based on
-   validation loss.

    Args:
        model_wrapper (ModelWrapper): The model to be fine-tuned.
        validation_loader (DataLoader): Data loader for the validation set,
            used by the scheduler.
        config_manager (ConfigManager): The simulation's configuration.
    """
    logger.info(f"Fine-tuning model for {config_manager.finetune_epochs} epoch(s) with scheduler...")
    _setup_and_run_training('finetune', model_wrapper, config_manager, validation_loader)
    model_wrapper.fitness_is_current = False
    logger.info("Fine-tuning complete.")
