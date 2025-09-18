import torch
import torch.optim as optim
import torch.nn.functional as F
from model import SimpleCNN
from data import get_mnist_dataloaders
import copy

class ModelWrapper:
    """
    A wrapper for a neural network model to hold its metadata, such as its
    specialized niche and its fitness score.
    """
    def __init__(self, niche_digits, device='cpu'):
        self.niche_digits = niche_digits
        self.device = device
        self.model = SimpleCNN().to(device)
        # Fitness is measured as accuracy on the full test set.
        self.fitness = 0.0

def specialize(model_wrapper, epochs=1):
    """
    Trains a model on its specialized data niche. This simulates the
    "resource competition" phase where models become experts in a specific area.
    """
    print(f"Specializing model on niche: {model_wrapper.niche_digits} for {epochs} epoch(s)...")
    # Get the dataloader for the model's specific niche
    train_loader, _ = get_mnist_dataloaders(niche_digits=model_wrapper.niche_digits)
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=0.001)

    model_wrapper.model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(model_wrapper.device), target.to(model_wrapper.device)
            optimizer.zero_grad()
            output = model_wrapper.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    print("Specialization complete.")

def evaluate(model_wrapper):
    """
    Evaluates the model's fitness on the *full* test dataset.
    The accuracy on the general task is our measure of fitness.
    """
    # We always evaluate on the full test set to measure general performance
    _, test_loader = get_mnist_dataloaders()
    model_wrapper.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model_wrapper.device), target.to(model_wrapper.device)
            output = model_wrapper.model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    model_wrapper.fitness = accuracy
    return accuracy

def select_mates(population):
    """
    Selects two parent models for mating based on their fitness.
    This is a simplified "intelligent mating" where we assume the two defined
    niches are complementary and we just need the best from each.
    """
    print("Selecting best models from each niche for mating...")
    # Niche 1: Digits 0-4
    parent1 = max(
        [m for m in population if m.niche_digits == list(range(5))],
        key=lambda m: m.fitness
    )
    # Niche 2: Digits 5-9
    parent2 = max(
        [m for m in population if m.niche_digits == list(range(5, 10))],
        key=lambda m: m.fitness
    )
    print(f"  - Parent 1 (Niche {parent1.niche_digits}, Fitness: {parent1.fitness:.2f}%)")
    print(f"  - Parent 2 (Niche {parent2.niche_digits}, Fitness: {parent2.fitness:.2f}%)")
    return parent1, parent2

def merge(parent1, parent2):
    """
    Merges two parent models to create a new child model by averaging their weights.
    This is a simplified version of the paper's "crossover".
    """
    print("Merging parent models to create child...")
    child_wrapper = ModelWrapper(niche_digits=list(range(10)), device=parent1.device)
    child_model_state_dict = child_wrapper.model.state_dict()

    parent1_state_dict = parent1.model.state_dict()
    parent2_state_dict = parent2.model.state_dict()

    for key in child_model_state_dict:
        # Average the parameters from both parents
        child_model_state_dict[key] = (parent1_state_dict[key] + parent2_state_dict[key]) / 2.0

    child_wrapper.model.load_state_dict(child_model_state_dict)
    print("Merging complete.")
    return child_wrapper

def mutate(model_wrapper, mutation_rate=0.01, mutation_strength=0.1):
    """
    Applies random mutations to the model's weights to maintain genetic diversity.
    """
    print("Mutating child model...")
    with torch.no_grad():
        for param in model_wrapper.model.parameters():
            if len(param.shape) > 1: # Mutate only multi-dimensional layers (conv, linear)
                # Create a random mask to decide which weights to mutate
                mutation_mask = (torch.rand(param.shape) < mutation_rate).to(model_wrapper.device)
                # Generate random noise to add to the weights
                mutation = torch.randn(param.shape).to(model_wrapper.device) * mutation_strength
                # Apply the mutation where the mask is True
                param.data += mutation * mutation_mask
    print("Mutation complete.")
    return model_wrapper

def finetune(model_wrapper, epochs=1):
    """
    Fine-tunes a model on the full dataset. This is useful for the child
    model to learn how to integrate the knowledge from its parents.
    """
    print(f"Fine-tuning model on the full dataset for {epochs} epoch(s)...")
    # Get the dataloader for the full dataset
    train_loader, _ = get_mnist_dataloaders()
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=0.001)

    model_wrapper.model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(model_wrapper.device), target.to(model_wrapper.device)
            optimizer.zero_grad()
            output = model_wrapper.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    print("Fine-tuning complete.")
