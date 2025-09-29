"""Handles the loading and preparation of image classification datasets.

This module provides a generic function to create PyTorch DataLoaders for
different datasets, such as CIFAR-10 and MNIST. It can be configured to
provide the full dataset, a niche subset for specialist training, or a
smaller random subset for rapid testing.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

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

def get_dataloaders(dataset_name='CIFAR10', batch_size=64, niche_classes=None, subset_percentage=1.0):
    """Creates and returns PyTorch DataLoaders for a specified dataset.

    This function prepares a dataset for training and testing. It can serve
    the full dataset, a "niche" subset of specific classes, or a random
    subset for rapid testing. It applies the appropriate transformations
    based on the dataset name.

    Args:
        dataset_name (str, optional): The name of the dataset to load.
            Supported options are 'CIFAR10', 'MNIST', and 'LLM'.
            Defaults to 'CIFAR10'.
        batch_size (int, optional): The number of samples per batch.
            Defaults to 64.
        niche_classes (list[int], optional): A list of class indices to
            exclusively include in the training set. Defaults to None.
        subset_percentage (float, optional): A float between 0.0 and 1.0
            specifying the fraction of the dataset to use. Defaults to 1.0.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and
            test DataLoaders.

    Raises:
        ValueError: If an unsupported `dataset_name` is provided.
    """
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'LLM':
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        raw_dataset = load_dataset('banking77')

        # Explicitly convert to list to ensure compatibility with the tokenizer
        train_texts = list(raw_dataset['train']['text'])
        train_labels = list(raw_dataset['train']['label'])
        test_texts = list(raw_dataset['test']['text'])
        test_labels = list(raw_dataset['test']['label'])

        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

        full_train_dataset = TextDataset(train_encodings, train_labels)
        full_test_dataset = TextDataset(test_encodings, test_labels)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please use 'CIFAR10', 'MNIST', or 'LLM'.")

    # Create a random subset if specified
    if subset_percentage < 1.0:
        num_train = int(len(full_train_dataset) * subset_percentage)
        train_indices = torch.randperm(len(full_train_dataset))[:num_train]
        full_train_dataset = Subset(full_train_dataset, train_indices)

        num_test = int(len(full_test_dataset) * subset_percentage)
        test_indices = torch.randperm(len(full_test_dataset))[:num_test]
        full_test_dataset = Subset(full_test_dataset, test_indices)

    if niche_classes is not None:
        if dataset_name == 'LLM':
            # For TextDataset, we need to access the label from the item dictionary
            train_indices = [i for i, item in enumerate(full_train_dataset) if item['labels'].item() in niche_classes]
        else:
            # For image datasets, the label is the second element of the tuple
            train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in niche_classes]

        niche_train_subset = Subset(full_train_dataset, train_indices)
        train_loader = DataLoader(dataset=niche_train_subset, batch_size=batch_size, shuffle=True)
        full_test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, full_test_loader

    train_loader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    print("--- Testing CIFAR-10 DataLoaders ---")
    cifar_train, cifar_test = get_dataloaders('CIFAR10')
    print(f"CIFAR-10 training samples: {len(cifar_train.dataset)}")
    print(f"CIFAR-10 test samples: {len(cifar_test.dataset)}")

    print("\n--- Testing MNIST DataLoaders ---")
    mnist_train, mnist_test = get_dataloaders('MNIST')
    print(f"MNIST training samples: {len(mnist_train.dataset)}")
    print(f"MNIST test samples: {len(mnist_test.dataset)}")

    print("\n--- Testing LLM (banking77) DataLoaders ---")
    llm_train, llm_test = get_dataloaders('LLM')
    print(f"LLM training samples: {len(llm_train.dataset)}")
    print(f"LLM test samples: {len(llm_test.dataset)}")

    print("\n--- Testing Niche Loader (LLM) ---")
    niche_train, niche_test = get_dataloaders('LLM', niche_classes=[0, 1, 2])
    print(f"Niche training samples: {len(niche_train.dataset)}")
    print(f"Niche test samples (full): {len(niche_test.dataset)}")