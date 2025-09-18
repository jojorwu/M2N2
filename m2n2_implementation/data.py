import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_dataloaders(batch_size=64, niche_digits=None):
    """
    Loads the MNIST dataset and returns DataLoaders for training and testing.
    If `niche_digits` is provided, it filters the dataset to only include those digits,
    creating a specialized "niche" for training.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the full training and test datasets
    full_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    full_test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    if niche_digits is not None:
        # Create a niche-specific training loader
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in niche_digits]
        niche_train_subset = Subset(full_train_dataset, train_indices)
        train_loader = DataLoader(dataset=niche_train_subset, batch_size=batch_size, shuffle=True)

        # For evaluation, we provide a loader for the full test set to see how the specialist performs overall
        full_test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, full_test_loader

    # If no niche is specified, return DataLoaders for the full dataset
    train_loader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    # Example of how to use the function to create niche dataloaders

    # Niche 1: Digits 0-4
    niche1_digits = list(range(5))
    train_loader_1, test_loader_1 = get_mnist_dataloaders(niche_digits=niche1_digits)
    print(f"Niche 1 (Digits {niche1_digits}):")
    print(f"  - Training samples: {len(train_loader_1.dataset)}")
    print(f"  - Test samples (full): {len(test_loader_1.dataset)}")

    # Niche 2: Digits 5-9
    niche2_digits = list(range(5, 10))
    train_loader_2, test_loader_2 = get_mnist_dataloaders(niche_digits=niche2_digits)
    print(f"Niche 2 (Digits {niche2_digits}):")
    print(f"  - Training samples: {len(train_loader_2.dataset)}")
    print(f"  - Test samples (full): {len(test_loader_2.dataset)}")

    # Full dataset
    full_train_loader, full_test_loader = get_mnist_dataloaders()
    print("Full Dataset:")
    print(f"  - Training samples: {len(full_train_loader.dataset)}")
    print(f"  - Test samples: {len(full_test_loader.dataset)}")
