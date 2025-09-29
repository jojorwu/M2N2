import pytest
import torch
import sys
import os

# Add the project root to the Python path to allow for imports from m2n2_implementation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2n2_implementation.model import CifarCNN, MnistCNN, LLMClassifier

BATCH_SIZE = 4

def test_cifar_cnn_creation_and_forward_pass():
    """Tests that the CifarCNN can be instantiated and can process a dummy batch."""
    model = CifarCNN()
    # Dummy input for CIFAR-10: (batch, channels, height, width)
    dummy_input = torch.randn(BATCH_SIZE, 3, 32, 32)
    output = model(dummy_input)

    assert output.shape == (BATCH_SIZE, model.num_classes), "CifarCNN output shape is incorrect."

def test_mnist_cnn_creation_and_forward_pass():
    """Tests that the MnistCNN can be instantiated and can process a dummy batch."""
    model = MnistCNN()
    # Dummy input for MNIST: (batch, channels, height, width)
    dummy_input = torch.randn(BATCH_SIZE, 1, 28, 28)
    output = model(dummy_input)

    assert output.shape == (BATCH_SIZE, model.num_classes), "MnistCNN output shape is incorrect."

def test_llm_classifier_creation_and_forward_pass():
    """Tests that the LLMClassifier can be instantiated and can process a dummy batch."""
    # The number of labels for the banking77 dataset
    num_labels = 77
    model = LLMClassifier(num_labels=num_labels)

    # Dummy input for a transformer model
    # Vocabulary size for distilbert-base-uncased is 30522
    # Sequence length is set to 64 as in the data loader
    dummy_input_ids = torch.randint(0, 30522, (BATCH_SIZE, 64))
    dummy_attention_mask = torch.ones(BATCH_SIZE, 64)

    output = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)

    assert output.shape == (BATCH_SIZE, num_labels), "LLMClassifier output shape is incorrect."
    assert model.num_classes == num_labels, "LLMClassifier num_classes attribute is incorrect."