"""
Handles the loading and preparation of image and text classification datasets.

This module provides a generic function to create PyTorch DataLoaders for
different datasets (CIFAR-10, MNIST, Banking77 for LLMs). It can be configured
to provide the full dataset, a niche subset for specialist training, or a smaller
random subset for rapid testing. It also partitions the training set to create
a validation set.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np
import logging
from .utils import set_seed
from .enums import DatasetName, ModelName
from .constants import LLM_CACHE_DIR
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger("M2N2_SIMULATOR")

class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for handling tokenized text data from Hugging Face.

    This class wraps around the tokenized outputs from a Hugging Face tokenizer
    (which are dictionary-like) and the corresponding labels, making them
    compatible with PyTorch's DataLoader.

    Attributes:
        encodings (Dict[str, Any]): A dictionary containing the tokenized
                                    input IDs, attention masks, etc.
        labels (List[int]): A list of integer labels for the text samples.
    """
    def __init__(self, encodings: Dict[str, Any], labels: List[int]):
        """
        Initializes the TextDataset.

        Args:
            encodings (Dict[str, Any]): The tokenized text data.
            labels (List[int]): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the tokenized
                                     input as tensors and the corresponding
                                     label as a tensor.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.labels)

def _load_full_datasets(dataset_name: DatasetName, model_name: ModelName) -> Tuple[Dataset, Dataset, int]:
    """
    Loads the full training and testing datasets from disk or downloads them.

    This helper function handles the logic for loading different types of
    datasets, applying the correct transformations based on the model
    architecture, and caching datasets (like for LLMs) to speed up
    subsequent runs.

    Args:
        dataset_name (DatasetName): The enum for the dataset to load.
        model_name (ModelName): The enum for the model, used to determine
                                appropriate data transformations.

    Returns:
        Tuple[Dataset, Dataset, int]: A tuple containing the full training
                                      dataset, the full test dataset, and the
                                      number of classes in the dataset.

    Raises:
        ValueError: If an unsupported dataset name is provided.
    """
    if dataset_name == DatasetName.CIFAR10:
        transform_list = []
        if model_name == ModelName.RESNET:
            transform_list.append(transforms.Resize((224, 224)))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform = transforms.Compose(transform_list)
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = len(full_train_dataset.classes)
    elif dataset_name == DatasetName.MNIST:
        transform_list = []
        if model_name == ModelName.RESNET:
            transform_list.append(transforms.Resize((224, 224)))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform = transforms.Compose(transform_list)
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = len(full_train_dataset.classes)
    elif dataset_name == DatasetName.LLM:
        train_cache_path = os.path.join(LLM_CACHE_DIR, 'cached_banking77_train.pt')
        test_cache_path = os.path.join(LLM_CACHE_DIR, 'cached_banking77_test.pt')

        raw_dataset = load_dataset('banking77')
        num_classes = raw_dataset['train'].features['label'].num_classes

        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            full_train_dataset = torch.load(train_cache_path)
            full_test_dataset = torch.load(test_cache_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            train_texts, train_labels = list(raw_dataset['train']['text']), list(raw_dataset['train']['label'])
            test_texts, test_labels = list(raw_dataset['test']['text']), list(raw_dataset['test']['label'])
            train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
            test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)
            full_train_dataset = TextDataset(train_encodings, train_labels)
            full_test_dataset = TextDataset(test_encodings, test_labels)
            os.makedirs(LLM_CACHE_DIR, exist_ok=True)
            torch.save(full_train_dataset, train_cache_path)
            torch.save(full_test_dataset, test_cache_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please use 'CIFAR10', 'MNIST', or 'LLM'.")
    return full_train_dataset, full_test_dataset, num_classes

def get_dataloaders(
    dataset_name: DatasetName,
    model_name: ModelName,
    batch_size: int = 64,
    niche_classes: Optional[List[int]] = None,
    subset_percentage: float = 1.0,
    validation_split: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Creates and returns PyTorch DataLoaders for a specified dataset.

    This function prepares a dataset for training and testing. It can serve
    the full dataset, a "niche" subset of specific classes, or a random
    subset for rapid testing. It also partitions the training set to create
    a validation loader, ensuring reproducibility through controlled random
    seeding.

    Args:
        dataset_name (DatasetName): The dataset to load (e.g., CIFAR10).
        model_name (ModelName): The model architecture, used to apply
            model-specific transforms.
        batch_size (int, optional): The number of samples per batch.
            Defaults to 64.
        niche_classes (Optional[List[int]], optional): A list of class indices
            to exclusively include in the training set for specialist training.
            If None, all classes are used. Defaults to None.
        subset_percentage (float, optional): A float between 0.0 and 1.0
            specifying the fraction of the dataset to use for faster runs.
            Defaults to 1.0.
        validation_split (float, optional): The proportion of the training
            set to hold out for validation. Defaults to 0.1.
        seed (Optional[int], optional): A random seed to ensure reproducible
            data splitting and shuffling. If None, a random seed is used.
            Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: A tuple containing the
            training DataLoader, validation DataLoader, test DataLoader, and the
            total number of classes in the dataset.
    """
    full_train_dataset, full_test_dataset, num_classes = _load_full_datasets(dataset_name, model_name)

    if niche_classes is not None:
        if dataset_name == DatasetName.LLM:
            niche_indices = [i for i, item in enumerate(full_train_dataset) if item['labels'].item() in niche_classes]
        else:
            targets = full_train_dataset.targets if hasattr(full_train_dataset, 'targets') else [label for _, label in full_train_dataset]
            niche_indices = [i for i, label in enumerate(targets) if label in niche_classes]
        full_train_dataset = Subset(full_train_dataset, niche_indices)

    if subset_percentage < 1.0:
        subset_g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        num_train = int(len(full_train_dataset) * subset_percentage)
        train_indices = torch.randperm(len(full_train_dataset), generator=subset_g)[:num_train].tolist()
        full_train_dataset = Subset(full_train_dataset, train_indices)
        num_test = int(len(full_test_dataset) * subset_percentage)
        test_indices = torch.randperm(len(full_test_dataset), generator=subset_g)[:num_test].tolist()
        full_test_dataset = Subset(full_test_dataset, test_indices)

    val_g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    num_train = len(full_train_dataset)
    split = int(np.floor(validation_split * num_train))
    if validation_split > 0 and split == 0:
        logger.warning(
            f"Validation split {validation_split} is too small for dataset "
            f"size {num_train}, resulting in an empty validation set."
        )
    train_subset, validation_subset = random_split(full_train_dataset, [num_train - split, split], generator=val_g)

    loader_g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, generator=loader_g, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=validation_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, validation_loader, test_loader, num_classes

if __name__ == '__main__':
    """
    Example script to demonstrate the functionality of the data loaders.

    When this script is run directly, it initializes and tests the data loaders
    for different datasets and prints the number of batches, serving as a quick
    verification of the data pipeline.
    """
    print("--- Testing DataLoaders with Validation Split ---")
    train_loader, val_loader, test_loader, num_classes_cifar = get_dataloaders(
        DatasetName.CIFAR10, ModelName.CNN, subset_percentage=0.1, seed=42
    )
    print(f"CIFAR-10 training batches: {len(train_loader)}")
    print(f"CIFAR-10 validation batches: {len(val_loader)}")
    print(f"CIFAR-10 test batches: {len(test_loader)}")
    print(f"CIFAR-10 number of classes: {num_classes_cifar}")

    train_loader, val_loader, test_loader, num_classes_llm = get_dataloaders(
        DatasetName.LLM, ModelName.LLM, subset_percentage=0.1, seed=42
    )
    print(f"\nLLM training batches: {len(train_loader)}")
    print(f"LLM validation batches: {len(val_loader)}")
    print(f"LLM test batches: {len(test_loader)}")
    print(f"LLM number of classes: {num_classes_llm}")
