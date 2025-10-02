"""Handles the loading and preparation of image classification datasets.

This module provides a generic function to create PyTorch DataLoaders for
different datasets. It can be configured to provide the full dataset, a
niche subset for specialist training, or a smaller random subset for
rapid testing. It can also partition the training set to create a
validation set.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np
from .utils import set_seed

class TextDataset(Dataset):
    """A custom PyTorch Dataset for handling tokenized text data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def _load_full_datasets(dataset_name, model_name):
    """Loads the full training and testing datasets based on the dataset name."""
    if dataset_name == 'CIFAR10':
        transform_list = []
        if model_name == 'RESNET':
            transform_list.append(transforms.Resize((224, 224)))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform = transforms.Compose(transform_list)
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        transform_list = []
        if model_name == 'RESNET':
            transform_list.append(transforms.Resize((224, 224)))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform = transforms.Compose(transform_list)
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'LLM':
        cache_dir = 'src/cache'
        train_cache_path = os.path.join(cache_dir, 'cached_banking77_train.pt')
        test_cache_path = os.path.join(cache_dir, 'cached_banking77_test.pt')
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            full_train_dataset = torch.load(train_cache_path)
            full_test_dataset = torch.load(test_cache_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            raw_dataset = load_dataset('banking77')
            train_texts, train_labels = list(raw_dataset['train']['text']), list(raw_dataset['train']['label'])
            test_texts, test_labels = list(raw_dataset['test']['text']), list(raw_dataset['test']['label'])
            train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
            test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)
            full_train_dataset = TextDataset(train_encodings, train_labels)
            full_test_dataset = TextDataset(test_encodings, test_labels)
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(full_train_dataset, train_cache_path)
            torch.save(full_test_dataset, test_cache_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please use 'CIFAR10', 'MNIST', or 'LLM'.")
    return full_train_dataset, full_test_dataset

def get_dataloaders(dataset_name='CIFAR10', model_name=None, batch_size=64, niche_classes=None, subset_percentage=1.0, validation_split=0.1, seed=None):
    """Creates and returns PyTorch DataLoaders for a specified dataset.

    This function prepares a dataset for training and testing. It can serve
    the full dataset, a "niche" subset of specific classes, or a random
    subset for rapid testing. It also partitions the training set to create
    a validation loader.

    Args:
        dataset_name (str, optional): The name of the dataset to load.
            Supported options are 'CIFAR10', 'MNIST', and 'LLM'.
            Defaults to 'CIFAR10'.
        model_name (str, optional): The name of the model architecture.
            Used to apply model-specific transforms (e.g., resizing for
            ResNet). Defaults to None.
        batch_size (int, optional): The number of samples per batch.
            Defaults to 64.
        niche_classes (list[int], optional): A list of class indices to
            exclusively include in the training set. Defaults to None.
        subset_percentage (float, optional): A float between 0.0 and 1.0
            specifying the fraction of the dataset to use. Defaults to 1.0.
        validation_split (float, optional): The proportion of the training
            set to use for validation. Defaults to 0.1.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the
            training, validation, and test DataLoaders.

    Raises:
        ValueError: If an unsupported `dataset_name` is provided.
    """
    if seed is not None:
        set_seed(seed)

    full_train_dataset, full_test_dataset = _load_full_datasets(dataset_name, model_name)

    if subset_percentage < 1.0:
        num_train = int(len(full_train_dataset) * subset_percentage)
        train_indices = np.random.permutation(len(full_train_dataset))[:num_train]
        full_train_dataset = Subset(full_train_dataset, train_indices)

        num_test = int(len(full_test_dataset) * subset_percentage)
        test_indices = np.random.permutation(len(full_test_dataset))[:num_test]
        full_test_dataset = Subset(full_test_dataset, test_indices)

    if niche_classes is not None:
        if dataset_name == 'LLM':
            niche_indices = [i for i, item in enumerate(full_train_dataset) if item['labels'].item() in niche_classes]
        else:
            niche_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in niche_classes]
        full_train_dataset = Subset(full_train_dataset, niche_indices)

    # Split training data into training and validation
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(full_train_dataset, train_idx)
    validation_subset = Subset(full_train_dataset, valid_idx)

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, generator=g)
    validation_loader = DataLoader(dataset=validation_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader

if __name__ == '__main__':
    print("--- Testing DataLoaders with Validation Split ---")
    train_loader, val_loader, test_loader = get_dataloaders('CIFAR10', subset_percentage=0.1)
    print(f"CIFAR-10 training batches: {len(train_loader)}")
    print(f"CIFAR-10 validation batches: {len(val_loader)}")
    print(f"CIFAR-10 test batches: {len(test_loader)}")

    train_loader, val_loader, test_loader = get_dataloaders('LLM', subset_percentage=0.1)
    print(f"\nLLM training batches: {len(train_loader)}")
    print(f"LLM validation batches: {len(val_loader)}")
    print(f"LLM test batches: {len(test_loader)}")