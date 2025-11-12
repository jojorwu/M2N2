"""Defines the neural network architectures used in the experiment.

This module contains the PyTorch `nn.Module` definitions for all models used
in the evolutionary simulation. This includes a simple Convolutional Neural
Network (CNN) for image tasks, a pre-trained ResNet model, and a
Transformer-based model for NLP tasks.
"""
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
from torchvision import models

class CifarCNN(nn.Module):
    """
    A simple Convolutional Neural Network designed for CIFAR-10 images.

    This architecture is tailored for 3x32x32 images. It consists of two
    convolutional layers, each followed by ReLU activation and max pooling,
    and then three fully connected layers for classification.

    Attributes:
        num_classes (int): The number of output classes for the final layer.
    """
    def __init__(self, num_classes: int = 10):
        """
        Initializes the layers of the CifarCNN model.

        Args:
            num_classes (int, optional): The number of output classes.
                Defaults to 10.
        """
        super(CifarCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the CifarCNN.

        The input tensor is passed through two convolutional blocks and then
        flattened before passing through three fully connected layers.

        Args:
            x (torch.Tensor): The input tensor, representing a batch of
                images with shape (N, 3, 32, 32).

        Returns:
            torch.Tensor: The output tensor of raw logits, with shape (N, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LLMClassifier(nn.Module):
    """
    A Transformer-based model for sequence classification using DistilBERT.

    This model leverages a pre-trained DistilBERT from Hugging Face and adds a
    classification head on top. It is designed for NLP tasks.

    Attributes:
        num_classes (int): The number of output classes for the classification head.
        bert (DistilBertForSequenceClassification): The underlying transformer model.
    """
    def __init__(self, num_labels: int):
        """
        Initializes the LLMClassifier model.

        Args:
            num_labels (int): The number of output classes for the
                classification head.
        """
        super(LLMClassifier, self).__init__()
        self.num_classes = num_labels
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=self.num_classes
        )

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the LLMClassifier.

        Args:
            input_ids (torch.Tensor): A tensor of token IDs of shape
                (batch_size, sequence_length).
            attention_mask (torch.Tensor): A tensor indicating which tokens
                should be attended to, of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The output logits from the model's classification
                head, of shape (batch_size, num_labels).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class ResNetClassifier(nn.Module):
    """
    A classifier based on a pre-trained ResNet-18 model.

    This model uses a pre-trained ResNet-18 from Torchvision and replaces
    the final fully connected layer to adapt it for a specific number of
    classes. This technique is a form of transfer learning.

    Attributes:
        num_classes (int): The number of output classes.
        resnet (torch.nn.Module): The underlying ResNet-18 model.
    """
    def __init__(self, num_classes: int = 10):
        """
        Initializes the ResNetClassifier model.

        Args:
            num_classes (int, optional): The number of output classes.
                Defaults to 10.
        """
        super(ResNetClassifier, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the ResNetClassifier.

        Args:
            x (torch.Tensor): The input tensor of images with shape (N, 3, H, W).
                For optimal ResNet performance, H and W should be >= 224.

        Returns:
            torch.Tensor: The output tensor of raw logits, with shape (N, num_classes).
        """
        return self.resnet(x)
