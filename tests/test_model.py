import unittest
import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import CifarCNN, LLMClassifier, ResNetClassifier

class TestModelArchitectures(unittest.TestCase):
    """Tests for the model architectures defined in model.py."""

    def test_cifarcnn_forward_pass(self):
        """Tests the forward pass of the CifarCNN model."""
        model = CifarCNN()
        dummy_input = torch.randn(4, 3, 32, 32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, 10), "CifarCNN output shape is incorrect!")

    def test_llmclassifier_forward_pass(self):
        """Tests the forward pass of the LLMClassifier model."""
        model = LLMClassifier(num_labels=77)
        dummy_input_ids = torch.randint(0, 30522, (4, 16))
        dummy_attention_mask = torch.ones(4, 16)
        output = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
        self.assertEqual(output.shape, (4, 77), "LLMClassifier output shape is incorrect!")

    def test_resnetclassifier_forward_pass(self):
        """Tests the forward pass of the ResNetClassifier model."""
        model = ResNetClassifier(num_classes=10)
        dummy_input = torch.randn(4, 3, 32, 32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, 10), "ResNetClassifier output shape is incorrect!")

if __name__ == '__main__':
    unittest.main()
