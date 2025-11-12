"""
Provides a factory function to create and return neural network models.

This module decouples the rest of the application from the concrete
implementation of model architectures. By using a factory, new models can be
added to the system with minimal changes to the core logic.
"""
from torch import nn
from .enums import ModelName
from .model import CifarCNN, LLMClassifier, ResNetClassifier

def create_model(model_name: ModelName, num_classes: int, device: str) -> nn.Module:
    """
    Instantiates and returns a model based on the provided model name.

    This factory function is the central point for creating all model instances
    in the simulation, ensuring that the correct model class is instantiated
    based on the configuration.

    Args:
        model_name (ModelName): The enum representing the desired model
            architecture (e.g., CNN, RESNET, LLM).
        num_classes (int): The number of output classes for the model's
            classification head.
        device (str): The device ('cpu' or 'cuda') to which the model's
            parameters will be moved.

    Returns:
        nn.Module: An instance of the specified model, moved to the correct device.

    Raises:
        ValueError: If the provided `model_name` is not a supported enum member.
    """
    if model_name == ModelName.CNN:
        return CifarCNN(num_classes=num_classes).to(device)
    elif model_name == ModelName.LLM:
        return LLMClassifier(num_labels=num_classes).to(device)
    elif model_name == ModelName.RESNET:
        return ResNetClassifier(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
