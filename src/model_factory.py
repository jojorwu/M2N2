"""
Provides a factory function to create and return neural network models.

This module decouples the rest of the application from the concrete
implementation of model architectures, allowing for easier extension and
maintenance.
"""
from torch import nn
from .enums import ModelName
from .model import CifarCNN, LLMClassifier, ResNetClassifier

def create_model(model_name: ModelName, num_classes: int, device: str) -> nn.Module:
    """
    Instantiates and returns a model based on the provided model name.

    Args:
        model_name (ModelName): The enum representing the desired model
            architecture.
        num_classes (int): The number of output classes for the model.
        device (str): The device to move the model to ('cpu' or 'cuda').

    Returns:
        nn.Module: An instance of the specified model, moved to the
            correct device.

    Raises:
        ValueError: If the provided model_name is not supported.
    """
    if model_name == ModelName.CIFAR10:
        return CifarCNN(num_classes=num_classes).to(device)
    elif model_name == ModelName.LLM:
        return LLMClassifier(num_labels=num_classes).to(device)
    elif model_name == ModelName.RESNET:
        return ResNetClassifier(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
