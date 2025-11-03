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
from .model import CifarCNN, LLMClassifier, ResNetClassifier
from .data import get_dataloaders
from .model_factory import create_model
from .merge_strategies import (
    MergeStrategy,
    AverageMergeStrategy,
    FitnessWeightedMergeStrategy,
    LayerWiseMergeStrategy,
    SequentialConstructiveMergeStrategy,
)
from .model_wrapper import ModelWrapper
from typing import List, Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader
from torch import nn

logger = logging.getLogger("M2N2_SIMULATOR")
import copy
import random
from tqdm import tqdm

def _run_training_epoch(model_wrapper: ModelWrapper, optimizer: optim.Optimizer, train_loader: DataLoader, scaler: torch.cuda.amp.GradScaler, precision: str, description: str, show_progress_bar: bool = True) -> float:
    """Runs a single training epoch for a given model and returns the average loss."""
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
    """A generalized helper to run a training session for a model."""
    if config_manager.precision_config == '64':
        model_wrapper.model.double()

    scaler = torch.cuda.amp.GradScaler(enabled=(config_manager.precision_config == '16' and 'cuda' in model_wrapper.device))

    try:
        for epoch in range(epochs):
            logger.info(f"  - Epoch {epoch + 1}/{epochs}")
            avg_train_loss = _run_training_epoch(
                model_wrapper,
                optimizer,
                train_loader,
                scaler,
                config_manager.precision_config,
                description,
                show_progress_bar=config_manager.show_progress_bar
            )
            if scheduler and validation_loader:
                avg_val_loss = _calculate_loss(model_wrapper, validation_loader)
                scheduler.step(avg_val_loss)
                logger.info(f"  - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
    except RuntimeError as e:
        logger.warning(
            f"A RuntimeError occurred during the training session for niche "
            f"{model_wrapper.niche_classes}: {e}. This might be due to CUDA "
            "out-of-memory. The session for this model will be aborted, "
            "but the simulation will continue."
        )

def _setup_and_run_training(
    mode: str,
    model_wrapper: ModelWrapper,
    config_manager: "ConfigManager",
    validation_loader: Optional[DataLoader] = None
) -> None:
    """A helper to set up and run a training session for either specialization or fine-tuning."""

    if mode == 'specialize':
        epochs = config_manager.specialize_epochs
        description = f"Specializing Niche {model_wrapper.niche_classes}"
        niche_classes = model_wrapper.niche_classes
        scheduler = None
    elif mode == 'finetune':
        epochs = config_manager.finetune_epochs
        description = "Fine-tuning Child"
        niche_classes = None
        if not validation_loader:
            raise ValueError("Validation loader is required for fine-tuning with a scheduler.")
    else:
        raise ValueError(f"Invalid mode for training session: {mode}")

    train_loader, _, _, _ = get_dataloaders(
        dataset_name=config_manager.dataset_name,
        model_name=model_wrapper.model_name,
        niche_classes=niche_classes,
        subset_percentage=config_manager.subset_percentage,
        seed=config_manager.seed,
        validation_split=0.0
    )

    if len(train_loader) == 0:
        logger.warning(
            f"Skipping {mode} for niche {model_wrapper.niche_classes} "
            "as the data loader is empty. This can happen with a small "
            "subset_percentage or if niche classes have no samples in the subset."
        )
        return

    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=config_manager.learning_rate)
    if mode == 'finetune':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config_manager.scheduler_patience, factor=config_manager.scheduler_factor)

    _run_training_session(
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        config_manager=config_manager,
        epochs=epochs,
        description=description,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_loader=validation_loader
    )

def specialize(model_wrapper: ModelWrapper, config_manager: "ConfigManager") -> None:
    """Trains a model in-place on its specialized data niche."""
    logger.info(f"Specializing model on niche {model_wrapper.niche_classes} for {config_manager.specialize_epochs} epoch(s) with {config_manager.precision_config}-bit precision...")
    _setup_and_run_training(
        mode='specialize',
        model_wrapper=model_wrapper,
        config_manager=config_manager
    )
    model_wrapper.fitness_is_current = False
    logger.info("Specialization complete.")

from .selection_strategies import MateSelectionStrategy, HealingMateSelectionStrategy
from .generation_strategies import GenerationStrategy, ReplaceWorstStrategy

def select_mates(
    population: List[ModelWrapper],
    num_pairs: int,
    strategy: "MateSelectionStrategy",
    config_manager: "ConfigManager"
) -> List[Tuple[ModelWrapper, ModelWrapper]]:
    """
    Selects multiple pairs of parents from the population using a specified strategy.
    """
    return strategy.select_parent_pairs(
        population,
        num_pairs=num_pairs,
        config_manager=config_manager
    )

def merge(parent1: ModelWrapper, parent2: ModelWrapper, strategy: "MergeStrategy", validation_loader: Optional[DataLoader] = None) -> ModelWrapper:
    """
    Merges two parent models into a new child model using a specified strategy.
    """
    logger.info(f"Merging parent models to create child using '{strategy.__class__.__name__}' strategy...")

    child_model_state_dict = strategy.merge(parent1, parent2, validation_loader)

    # Create and return the new child model
    num_classes = parent1.model.num_classes
    child_model = create_model(parent1.model_name, num_classes, parent1.device)
    child_model.load_state_dict(child_model_state_dict)
    child_wrapper = ModelWrapper(
        model_name=parent1.model_name,
        model=child_model,
        niche_classes=list(range(num_classes)),
        device=parent1.device
    )
    logger.info("Merging complete.")
    return child_wrapper

def mutate(model_wrapper: ModelWrapper, generation: int, config_manager: "ConfigManager", seed: Optional[int] = None) -> ModelWrapper:
    """
    Applies deterministic, adaptively scaled Gaussian mutations to a model.

    This function introduces genetic diversity by altering a fixed fraction of
    the model's weights. The process is made deterministic by using a seeded
    PyTorch generator. This implementation is optimized for performance by
    creating a single generator and re-seeding it for each layer, which
    preserves layer-independent randomness while avoiding object creation
    overhead in the loop.

    Args:
        model_wrapper: The model to mutate.
        generation: The current generation number, for adaptive strength.
        config_manager: The configuration manager.
        seed: An optional seed for the random number generator to ensure
            reproducibility.

    Returns:
        The mutated model wrapper.
    """
    decayed_strength = config_manager.initial_mutation_strength * (config_manager.mutation_decay_factor ** generation)
    logger.info(f"Mutating child model (Gen: {generation}, Strength: {decayed_strength:.4f})...")

    generator = torch.Generator(device=model_wrapper.device)

    with torch.no_grad():
        param_idx = 0
        for param in model_wrapper.model.parameters():
            if param.dim() > 1:
                if seed is not None:
                    # Re-seed the generator for each layer to ensure independent,
                    # deterministic mutations for each parameter.
                    generator.manual_seed(seed + param_idx)

                num_weights = param.numel()
                num_to_mutate = int(num_weights * config_manager.mutation_rate)

                if num_to_mutate == 0:
                    continue

                indices_to_mutate = torch.randperm(num_weights, device=model_wrapper.device, generator=generator)[:num_to_mutate]
                mutation = torch.randn(num_to_mutate, device=model_wrapper.device, generator=generator) * decayed_strength

                param.view(-1)[indices_to_mutate] += mutation
            param_idx += 1

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
    Creates the next generation's population using a specified strategy.
    """
    return strategy.create_next_generation(
        current_population,
        offspring_pool,
        config_manager=config_manager
    )

def _calculate_loss(model_wrapper: ModelWrapper, data_loader: DataLoader) -> float:
    """A generic helper to calculate loss on a given data loader."""
    model_wrapper.model.eval()
    total_loss = 0.0
    device = model_wrapper.device

    with torch.no_grad():
        for batch in data_loader:
            output, target = model_wrapper._process_batch(batch)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()

    if len(data_loader) == 0:
        return 0.0
    return total_loss / len(data_loader)


def finetune(model_wrapper: ModelWrapper, validation_loader: DataLoader, config_manager: "ConfigManager") -> None:
    """Fine-tunes a model in-place on the full dataset with a scheduler."""
    logger.info(f"Fine-tuning model for {config_manager.finetune_epochs} epoch(s) with {config_manager.precision_config}-bit precision and ReduceLROnPlateau scheduler...")
    _setup_and_run_training(
        mode='finetune',
        model_wrapper=model_wrapper,
        config_manager=config_manager,
        validation_loader=validation_loader
    )
    model_wrapper.fitness_is_current = False
    logger.info("Fine-tuning complete.")