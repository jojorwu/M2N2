"""Defines the neural network architectures used in the experiment.

This module contains the definitions for all models used in the evolutionary
simulation, including Convolutional Neural Networks (CNNs) for image tasks
and a Transformer-based model for NLP tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification

class CifarCNN(nn.Module):
    """A Convolutional Neural Network designed for CIFAR-10 images.

    This architecture is tailored for 3x32x32 images. It consists of two
    convolutional layers, each followed by ReLU activation and max pooling,
    and then three fully connected layers for classification.

    Attributes:
        num_classes (int): The number of output classes.
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
        self.num_classes = 10
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # -> 32x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)      # -> 32x16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)# -> 64x16x16
        # The second pool operation reduces it to 64x8x8

        # Flattened size is 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes) # 10 output classes for CIFAR-10

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
        num_classes (int): The number of output classes.
        conv1 (nn.Conv2d): First convolutional layer (1 -> 32 channels).
        conv2 (nn.Conv2d): Second convolutional layer (32 -> 64 channels).
        pool (nn.MaxPool2d): Max pooling layer with a 2x2 kernel.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Output layer (10 classes).
    """
    def __init__(self):
        """Initializes the layers of the MnistCNN model."""
        super(MnistCNN, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flattened size is 64 channels * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

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

class LLMClassifier(nn.Module):
    """A Transformer-based model for sequence classification.

    This model uses a pre-trained DistilBERT from Hugging Face with a
    classification head on top. It is designed for NLP tasks like
    intent classification.

    Attributes:
        num_classes (int): The number of output classes.
        bert (DistilBertForSequenceClassification): The underlying
            transformer model.
    """
    def __init__(self, num_labels=77):
        """Initializes the LLMClassifier model.

        Args:
            num_labels (int, optional): The number of output classes for
                the classification head. Defaults to 77 for the
                banking77 dataset.
        """
        super(LLMClassifier, self).__init__()
        self.num_classes = num_labels
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=self.num_classes
        )

    def forward(self, input_ids, attention_mask):
        """Defines the forward pass of the LLMClassifier.

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

if __name__ == '__main__':
    # Test the CNN models
    print("--- CifarCNN Model Test ---")
    cifar_model = CifarCNN()
    cifar_dummy_input = torch.randn(4, 3, 32, 32)
    cifar_output = cifar_model(cifar_dummy_input)
    print(f"CIFAR Dummy input shape: {cifar_dummy_input.shape}")
    print(f"CIFAR Output shape: {cifar_output.shape}")
    assert cifar_output.shape == (4, 10), "CIFAR CNN output shape is incorrect!"
    print("CifarCNN forward pass test successful!\n")

    # Test the LLM model
    print("--- LLMClassifier Model Test ---")
    llm_model = LLMClassifier(num_labels=77)
    # Batch of 4, sequence length of 16
    llm_dummy_input_ids = torch.randint(0, 30522, (4, 16))
    llm_dummy_attention_mask = torch.ones(4, 16)
    llm_output = llm_model(input_ids=llm_dummy_input_ids, attention_mask=llm_dummy_attention_mask)
    print(f"LLM Dummy input shape: {llm_dummy_input_ids.shape}")
    print(f"LLM Output shape: {llm_output.shape}")
    assert llm_output.shape == (4, 77), "LLMClassifier output shape is incorrect!"
    print("LLMClassifier forward pass test successful!")