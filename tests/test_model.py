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

import shutil
from unittest.mock import patch
from src.model_wrapper import ModelWrapper
from src.enums import ModelName

class TestModelWrapper(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_model_test"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_from_file_handles_corrupted_model(self):
        """
        Tests that ModelWrapper.from_file returns None and logs a warning when
        it encounters a corrupted or invalid model file.
        """
        # Create a dummy filename that matches the expected pattern
        filename = "model_niche_0_fitness_0.0.pth"
        filepath = os.path.join(self.test_dir, filename)

        # Write invalid content to the file
        with open(filepath, "w") as f:
            f.write("This is not a valid model file.")

        # Suppress logger setup to not interfere with assertLogs
        with patch('src.model_wrapper.logger') as mock_logger:
             # Attempt to load the corrupted model
            wrapper = ModelWrapper.from_file(
                filepath=filepath,
                model_name=ModelName.CNN,
                num_classes=10,
                device='cpu'
            )

            # Assert that the method returns None and logs a warning
            self.assertIsNone(wrapper, "ModelWrapper.from_file should return None for a corrupted file.")
            mock_logger.warning.assert_called_once()
            self.assertIn("Failed to load model", mock_logger.warning.call_args[0][0])


if __name__ == '__main__':
    unittest.main()
