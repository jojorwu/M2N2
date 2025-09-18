import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_dataloaders(batch_size=64, niche_classes=None):
    """
    Loads the CIFAR-10 dataset and returns DataLoaders for training and testing.
    If `niche_classes` is provided, it filters the dataset to only include those classes,
    creating a specialized "niche" for training.
    """
    # Transform for CIFAR-10, including normalization for 3-channel color images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the full training and test datasets
    full_train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    full_test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    if niche_classes is not None:
        # Create a niche-specific training loader
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in niche_classes]
        niche_train_subset = Subset(full_train_dataset, train_indices)
        train_loader = DataLoader(dataset=niche_train_subset, batch_size=batch_size, shuffle=True)

        # For evaluation, provide a loader for the full test set
        full_test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, full_test_loader

    # If no niche is specified, return DataLoaders for the full dataset
    train_loader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    # Example of how to use the function to create niche dataloaders for CIFAR-10
    # CIFAR-10 classes: 0:airplane, 1:automobile, 2:bird, 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck

    # Niche 1: Animals
    animal_classes = [2, 3, 4, 5, 6, 7]
    train_loader_1, test_loader_1 = get_cifar10_dataloaders(niche_classes=animal_classes)
    print(f"Niche 1 (Animals):")
    print(f"  - Training samples: {len(train_loader_1.dataset)}")
    print(f"  - Test samples (full): {len(test_loader_1.dataset)}")

    # Niche 2: Vehicles
    vehicle_classes = [0, 1, 8, 9]
    train_loader_2, test_loader_2 = get_cifar10_dataloaders(niche_classes=vehicle_classes)
    print(f"Niche 2 (Vehicles):")
    print(f"  - Training samples: {len(train_loader_2.dataset)}")
    print(f"  - Test samples (full): {len(test_loader_2.dataset)}")

    # Full dataset
    full_train_loader, full_test_loader = get_cifar10_dataloaders()
    print("Full Dataset:")
    print(f"  - Training samples: {len(full_train_loader.dataset)}")
    print(f"  - Test samples: {len(full_test_loader.dataset)}")
