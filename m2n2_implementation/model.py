import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer: input features 64 * 7 * 7 (from pooled conv2), 128 output features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer: 128 input features, 10 output features (for digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply convolution, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the image tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        # Apply fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply output layer
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # This block will run only when the script is executed directly
    # It's a good practice to test the model architecture here

    # Instantiate the model
    model = SimpleCNN()
    print("Model Architecture:")
    print(model)

    # Create a dummy input tensor to verify the forward pass
    # Batch size = 64, 1 color channel, 28x28 image size
    dummy_input = torch.randn(64, 1, 28, 28)
    output = model(dummy_input)

    print("\n--- Model Test ---")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # The output shape should be (batch_size, num_classes)
    assert output.shape == (64, 10), "The output shape is incorrect!"

    print("\nModel forward pass test successful!")
