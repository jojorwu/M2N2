import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarCNN(nn.Module):
    """A Convolutional Neural Network designed for CIFAR-10 images.

    This model architecture is tailored for 32x32x3 images. It consists of
    two convolutional layers followed by max pooling, and three fully
    connected layers for classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (3 -> 32 channels).
        conv2 (nn.Conv2d): Second convolutional layer (32 -> 64 channels).
        pool (nn.MaxPool2d): Max pooling layer with a 2x2 kernel.
        fc1 (nn.Linear): First fully connected layer (4096 -> 512).
        fc2 (nn.Linear): Second fully connected layer (512 -> 128).
        fc3 (nn.Linear): Output layer (128 -> 10 classes).
    """
    def __init__(self):
        super(CifarCNN, self).__init__()
        # Input channels = 3 (for color images), output channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # The input image size is 32x32. After two max pooling layers (32->16->8),
        # the image size is 8x8. The number of channels is 64 from conv2.
        # So, the flattened size is 64 * 8 * 8 = 4096.
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10) # 10 output classes for CIFAR-10

    def forward(self, x):
        """Defines the forward pass of the CifarCNN.

        Args:
            x (torch.Tensor): The input tensor of shape (N, 3, 32, 32),
              where N is the batch size.

        Returns:
            torch.Tensor: The output tensor of shape (N, 10), containing
              the raw logits for each of the 10 classes.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Keep the old MNIST CNN for reference, but rename it.
class MnistCNN(nn.Module):
    """A simple Convolutional Neural Network for MNIST classification.

    This model is designed for 28x28x1 grayscale images from the MNIST
    dataset. It is kept for reference purposes.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (1 -> 32 channels).
        conv2 (nn.Conv2d): Second convolutional layer (32 -> 64 channels).
        pool (nn.MaxPool2d): Max pooling layer with a 2x2 kernel.
        fc1 (nn.Linear): First fully connected layer (3136 -> 128).
        fc2 (nn.Linear): Output layer (128 -> 10 classes).
    """
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Defines the forward pass of the MnistCNN.

        Args:
            x (torch.Tensor): The input tensor of shape (N, 1, 28, 28),
              where N is the batch size.

        Returns:
            torch.Tensor: The output tensor of shape (N, 10), containing
              the raw logits for each of the 10 classes.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
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
