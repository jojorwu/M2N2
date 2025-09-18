import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_dataloaders(batch_size=64, niche_classes=None, subset_percentage=1.0):
    """Loads the CIFAR-10 dataset and returns DataLoaders.

    This function can be configured to load the full dataset, a specific "niche"
    of classes, or a random subset of the data for faster execution.

    Args:
        batch_size (int, optional): The number of samples per batch.
            Defaults to 64.
        niche_classes (list[int], optional): A list of class indices to
            include in the training set. If None, all classes are used.
            For CIFAR-10, classes are 0-9. Defaults to None.
        subset_percentage (float, optional): A value between 0.0 and 1.0
            that specifies the percentage of the dataset to use.
            Helpful for quick testing. Defaults to 1.0 (full dataset).

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set. If
              `niche_classes` is specified, this loader contains only data
              from those classes.
            - test_loader (DataLoader): DataLoader for the test set. This
              loader always contains data from all classes to ensure
              evaluation is performed on the general task.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create a random subset if specified
    if subset_percentage < 1.0:
        num_train = int(len(full_train_dataset) * subset_percentage)
        train_indices = torch.randperm(len(full_train_dataset))[:num_train]
        full_train_dataset = Subset(full_train_dataset, train_indices)

        num_test = int(len(full_test_dataset) * subset_percentage)
        test_indices = torch.randperm(len(full_test_dataset))[:num_test]
        full_test_dataset = Subset(full_test_dataset, test_indices)

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
