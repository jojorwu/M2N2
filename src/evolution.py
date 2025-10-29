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
        if show_progress_bar:
            data_iterator.set_postfix({'train_loss': f"{loss.item():.4f}"})

    return total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

def _run_training_session(
    model_wrapper: ModelWrapper,
    train_loader: DataLoader,
    epochs: int,
    precision: str,
    learning_rate: float,
    description: str,
    show_progress_bar: bool,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None,
    validation_loader: Optional[DataLoader] = None
) -> None:
    """A generalized helper to run a training session for a model."""
    if optimizer is None:
        optimizer = optim.Adam(model_wrapper.model.parameters(), lr=learning_rate)

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
            description,
            show_progress_bar=show_progress_bar
        )
        if scheduler and validation_loader:
            avg_val_loss = _calculate_loss(model_wrapper, validation_loader)
            scheduler.step(avg_val_loss)
            logger.info(f"  - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

def specialize(model_wrapper: ModelWrapper, config_manager: "ConfigManager") -> None:
    """Trains a model in-place on its specialized data niche."""
    logger.info(f"Specializing model on niche {model_wrapper.niche_classes} for {config_manager.specialize_epochs} epoch(s) with {config_manager.precision_config}-bit precision...")

    train_loader, _, _, _ = get_dataloaders(
        dataset_name=config_manager.dataset_name,
        model_name=model_wrapper.model_name,
        niche_classes=model_wrapper.niche_classes,
        subset_percentage=config_manager.subset_percentage,
        seed=config_manager.seed
    )

    _run_training_session(
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        epochs=config_manager.specialize_epochs,
        precision=config_manager.precision_config,
        learning_rate=config_manager.learning_rate,
        description=f"Specializing Niche {model_wrapper.niche_classes}",
        show_progress_bar=config_manager.show_progress_bar
    )

    model_wrapper.fitness_is_current = False
    logger.info("Specialization complete.")




from .selection_strategies import MateSelectionStrategy, HealingMateSelectionStrategy
from .generation_strategies import GenerationStrategy, ReplaceWorstStrategy

def select_mates(
    population: List[ModelWrapper],
    strategy: "MateSelectionStrategy",
    dataset_name: "DatasetName",
    subset_percentage: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[Optional[ModelWrapper], Optional[ModelWrapper]]:
    """
    Selects a pair of parents from the population using a specified strategy.
    """
    return strategy.select_mates(
        population,
        dataset_name=dataset_name,
        subset_percentage=subset_percentage,
        seed=seed
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

def mutate(model_wrapper: ModelWrapper, generation: int, config_manager: "ConfigManager") -> ModelWrapper:
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
        config_manager (ConfigManager): The configuration manager.

    Returns:
        ModelWrapper: The same model wrapper that was passed in, allowing
            for method chaining.
    """
    # Calculate the decayed mutation strength for the current generation
    decayed_strength = config_manager.initial_mutation_strength * (config_manager.mutation_decay_factor ** generation)
    logger.info(f"Mutating child model (Gen: {generation}, Strength: {decayed_strength:.4f})...")

    with torch.no_grad():
        for param in model_wrapper.model.parameters():
            if len(param.shape) > 1: # Mutate only multi-dimensional layers (conv, linear)
                # Create a random mask to decide which weights to mutate
                mutation_mask = (torch.rand(param.shape) < config_manager.mutation_rate).to(model_wrapper.device)
                # Generate random noise scaled by the decayed strength
                mutation = torch.randn(param.shape).to(model_wrapper.device) * decayed_strength
                # Apply the mutation where the mask is True
                param.data += mutation * mutation_mask
    # Mark fitness as not current, as the model has been modified.
    model_wrapper.fitness_is_current = False
    logger.info("Mutation complete.")
    return model_wrapper

def create_next_generation(
    current_population: List[ModelWrapper],
    new_child: ModelWrapper,
    population_size: int,
    dataset_name: "DatasetName",
    strategy: "GenerationStrategy",
    seed: Optional[int] = None
) -> List[ModelWrapper]:
    """
    Creates the next generation's population using a specified strategy.
    """
    return strategy.create_next_generation(
        current_population,
        new_child,
        population_size,
        dataset_name,
        seed=seed
    )

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


def finetune(model_wrapper: ModelWrapper, validation_loader: DataLoader, config_manager: "ConfigManager") -> None:
    """Fine-tunes a model in-place on the full dataset with a scheduler."""
    logger.info(f"Fine-tuning model for {config_manager.finetune_epochs} epoch(s) with {config_manager.precision_config}-bit precision and ReduceLROnPlateau scheduler...")

    train_loader, _, _, _ = get_dataloaders(
        dataset_name=config_manager.dataset_name,
        model_name=model_wrapper.model_name,
        subset_percentage=config_manager.subset_percentage,
        seed=config_manager.seed,
        validation_split=0.0
    )

    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=config_manager.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config_manager.scheduler_patience, factor=config_manager.scheduler_factor)

    _run_training_session(
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        epochs=config_manager.finetune_epochs,
        precision=config_manager.precision_config,
        learning_rate=config_manager.learning_rate,
        description="Fine-tuning Child",
        show_progress_bar=config_manager.show_progress_bar,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_loader=validation_loader
    )

    model_wrapper.fitness_is_current = False
    logger.info("Fine-tuning complete.")