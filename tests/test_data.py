import pytest
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the project root to the Python path to allow for imports from m2n2_implementation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2n2_implementation.data import get_dataloaders

# Use a very small subset for all tests to ensure they run quickly
SUBSET_PERCENTAGE = 0.01
BATCH_SIZE = 2

@pytest.mark.parametrize("dataset_name, expected_data_shape, expected_target_shape", [
    ("CIFAR10", (BATCH_SIZE, 3, 32, 32), (BATCH_SIZE,)),
    ("MNIST", (BATCH_SIZE, 1, 28, 28), (BATCH_SIZE,)),
])
def test_get_image_dataloaders(dataset_name, expected_data_shape, expected_target_shape):
    """Tests that get_dataloaders correctly loads image datasets."""
    train_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=BATCH_SIZE,
        subset_percentage=SUBSET_PERCENTAGE
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Check a batch from the train loader
    data, target = next(iter(train_loader))
    assert data.shape == expected_data_shape
    assert target.shape == expected_target_shape

def test_get_llm_dataloader():
    """Tests that get_dataloaders correctly loads and tokenizes the LLM dataset."""
    train_loader, test_loader = get_dataloaders(
        dataset_name='LLM',
        batch_size=BATCH_SIZE,
        subset_percentage=SUBSET_PERCENTAGE
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Check a batch from the train loader
    batch = next(iter(train_loader))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch

    # Check the shape of the tensors in the batch
    assert batch['input_ids'].shape == (BATCH_SIZE, 64) # batch_size x max_length
    assert batch['attention_mask'].shape == (BATCH_SIZE, 64)
    assert batch['labels'].shape == (BATCH_SIZE,)

def test_get_dataloaders_niche_selection():
    """Tests that the niche_classes argument correctly filters the dataset."""
    niche_classes = [3, 5] # cat and dog in CIFAR-10
    train_loader, _ = get_dataloaders(
        dataset_name='CIFAR10',
        niche_classes=niche_classes,
        subset_percentage=0.1 # Use a slightly larger subset to ensure we get samples
    )

    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    # Check that only labels from the specified niche are present
    assert set(all_labels).issubset(set(niche_classes))
    # Check that we actually got some labels, otherwise the test is meaningless
    assert len(all_labels) > 0

def test_unsupported_dataset_raises_error():
    """Tests that get_dataloaders raises a ValueError for an unsupported dataset name."""
    with pytest.raises(ValueError, match="Unsupported dataset: FAKE_DATASET"):
        get_dataloaders(dataset_name='FAKE_DATASET')