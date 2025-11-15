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


from typing import Dict

def _get_validation_fitness(model_wrapper: ModelWrapper, validation_loader: DataLoader, batch: Optional[Any] = None, model_state_dict: Optional[Dict[str, torch.Tensor]] = None) -> float:
    """
    Calculates a fitness score using a provided validation loader or a single
    batch. Can optionally load a new state dict into the model before
    evaluation, which is useful for evaluating temporary or hybrid models
    without creating a new ModelWrapper instance.
    """
    # If a state dict is provided, load it into the model for this evaluation.
    # This is a temporary change for the scope of this function.
    if model_state_dict:
        original_state_dict = model_wrapper.model.state_dict()
        model_wrapper.model.load_state_dict(model_state_dict)

    accuracy = _calculate_accuracy(model_wrapper, validation_loader, batch=batch)

    # If we loaded a temporary state dict, restore the original one.
    if model_state_dict:
        model_wrapper.model.load_state_dict(original_state_dict)

    return accuracy


from typing import Tuple, List

def _calculate_metrics(model_wrapper: ModelWrapper, data_loader: DataLoader) -> Tuple[float, List[float]]:
    """
    Calculates both overall and per-class accuracy in a single pass.
    """
    model_wrapper.model.eval()
    device = model_wrapper.device
    num_classes = model_wrapper.model.num_classes

    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch in data_loader:
            if model_wrapper.model_name == 'LLM':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target = batch['labels'].to(device)
                output = model_wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                data, target = batch
                data, target = data.to(device), target.to(device)
                if next(model_wrapper.model.parameters()).dtype == torch.float64:
                    data = data.double()
                output = model_wrapper.model(data)

            _, predicted = torch.max(output, 1)
            correct_mask = (predicted == target)

            # Update totals for overall accuracy
            total_samples += target.size(0)
            total_correct += correct_mask.sum().item()

            # Update totals for per-class accuracy
            class_total += torch.bincount(target, minlength=num_classes)
            class_correct += torch.bincount(target[correct_mask], minlength=num_classes)

    # Calculate overall accuracy
    overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0.0

    # Calculate per-class accuracies
    class_total_cpu = class_total.cpu()
    class_correct_cpu = class_correct.cpu()
    valid_mask = class_total_cpu > 0
    class_accuracies = torch.zeros(num_classes)
    class_accuracies[valid_mask] = 100 * class_correct_cpu[valid_mask] / class_total_cpu[valid_mask]

    return overall_accuracy, class_accuracies.tolist()
