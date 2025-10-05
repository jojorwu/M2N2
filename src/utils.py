import random
import numpy as np
import torch
from typing import Optional, Any

def set_seed(seed: int) -> None:
    """Sets the seed for random number generators in random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The following two lines are for ensuring reproducible results on CUDA.
        # They can have a performance impact, so they are often disabled.
        # For this project, reproducibility is more important.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


from torch.utils.data import DataLoader
from .model_wrapper import ModelWrapper

def _calculate_accuracy(model_wrapper: ModelWrapper, data_loader: DataLoader, batch: Optional[Any] = None) -> float:
    """
    A generic helper to calculate accuracy on a given data loader or a single
    batch.
    """
    model_wrapper.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # If a single batch is provided, wrap it in a list to make it iterable
        data_source = [batch] if batch else data_loader

        for b in data_source:
            if model_wrapper.model_name == 'LLM':
                input_ids = b['input_ids'].to(model_wrapper.device)
                attention_mask = b['attention_mask'].to(model_wrapper.device)
                labels = b['labels'].to(model_wrapper.device)
                outputs = model_wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                data, target = b
                data = data.to(model_wrapper.device)
                target = target.to(model_wrapper.device)
                if next(model_wrapper.model.parameters()).dtype == torch.float64:
                    data = data.double()
                output = model_wrapper.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

    return 100 * correct / total if total > 0 else 0.0


def _get_validation_fitness(model_wrapper: ModelWrapper, validation_loader: DataLoader, batch: Optional[Any] = None) -> float:
    """
    Calculates a fitness score using a provided validation loader or a single
    batch.
    """
    return _calculate_accuracy(model_wrapper, validation_loader, batch=batch)
