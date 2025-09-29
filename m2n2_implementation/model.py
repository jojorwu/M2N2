"""Defines the neural network architectures used in the experiment.

This module contains the definitions for the Convolutional Neural Network (CNN)
models, conforming to Google's Python docstring style. The primary model is
`CifarCNN`, designed for the CIFAR-10 dataset, and a simpler `MnistCNN`
is included for reference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarCNN(nn.Module):
    """A Convolutional Neural Network designed for CIFAR-10 images.

    This architecture is tailored for 3x32x32 images. It consists of two
    convolutional layers, each followed by ReLU activation and max pooling,
    and then three fully connected layers for classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (3 input channels, 32
            output channels).
        conv2 (nn.Conv2d): Second convolutional layer (32 input channels, 64
            output channels).
        pool (nn.MaxPool2d): Max pooling layer with a 2x2 kernel and stride.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): The final output layer (10 classes for CIFAR-10).
    """
    def __init__(self):
        """Initializes the layers of the CifarCNN model."""
        super(CifarCNN, self).__init__()
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # -> 32x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)      # -> 32x16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)# -> 64x16x16
        # The second pool operation reduces it to 64x8x8

        # Flattened size is 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10) # 10 output classes for CIFAR-10

    def forward(self, x):
        """Defines the forward pass of the CifarCNN.

        The input tensor is passed through two convolutional blocks
        (conv -> relu -> pool) and then flattened before passing through
        three fully connected layers.

        Args:
            x (torch.Tensor): The input tensor, representing a batch of
                images with shape (N, 3, 32, 32), where N is the
                batch size.

        Returns:
            torch.Tensor: The output tensor of raw logits, with shape
                (N, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flatten the tensor for the FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Keep the old MNIST CNN for reference, but rename it.
class MnistCNN(nn.Module):
    """A simple Convolutional Neural Network for MNIST classification.

    This model is designed for 1x28x28 grayscale images from the MNIST
    dataset. It is included for reference and is not used in the main
    CIFAR-10 experiment.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (1 -> 32 channels).
        conv2 (nn.Conv2d): Second convolutional layer (32 -> 64 channels).
        pool (nn.MaxPool2d): Max pooling layer with a 2x2 kernel.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Output layer (10 classes).
    """
    def __init__(self):
        """Initializes the layers of the MnistCNN model."""
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flattened size is 64 channels * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Defines the forward pass of the MnistCNN.

        Args:
            x (torch.Tensor): The input tensor of shape (N, 1, 28, 28),
              where N is the batch size.

        Returns:
            torch.Tensor: The output tensor of raw logits, with shape
                (N, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Test the new CifarCNN model with a dummy input
    model = CifarCNN()
    print("Model Architecture (CifarCNN):")
    print(model)

    # Create a dummy input tensor to check the forward pass
    # Batch of 64 images, 3 color channels, 32x32 size
    dummy_input = torch.randn(64, 3, 32, 32)
    output = model(dummy_input)

    print("\n--- CifarCNN Model Test ---")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (64, 10), "The output shape is incorrect!"
    print("\nModel forward pass test successful!")