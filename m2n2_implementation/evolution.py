import torch
import torch.optim as optim
import torch.nn.functional as F
from model import CifarCNN
from data import get_cifar10_dataloaders
import copy

class ModelWrapper:
    """
    A wrapper for a neural network model to hold its metadata, such as its
    specialized niche and its fitness score.
    """
    def __init__(self, niche_classes, device='cpu'):
        self.niche_classes = niche_classes
        self.device = device
        self.model = CifarCNN().to(device)
        # Fitness is measured as accuracy on the full test set.
        self.fitness = 0.0

def specialize(model_wrapper, epochs=1):
    """
    Trains a model on its specialized data niche. This simulates the
    "resource competition" phase where models become experts in a specific area.
    """
    print(f"Specializing model on niche: {model_wrapper.niche_classes} for {epochs} epoch(s)...")
    # Get the dataloader for the model's specific niche
    train_loader, _ = get_cifar10_dataloaders(niche_classes=model_wrapper.niche_classes)
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
    _, test_loader = get_cifar10_dataloaders()
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

def select_mates(population, niche1_classes, niche2_classes):
    """
    Selects two parent models for mating from two complementary niches.
    Handles cases where one niche might be empty to prevent crashing.
    """
    print("Selecting best models from each niche for mating...")
    parent1, parent2 = None, None

    niche1_candidates = [m for m in population if m.niche_classes == niche1_classes]
    niche2_candidates = [m for m in population if m.niche_classes == niche2_classes]

    if niche1_candidates:
        parent1 = max(niche1_candidates, key=lambda m: m.fitness)

    if niche2_candidates:
        parent2 = max(niche2_candidates, key=lambda m: m.fitness)

    if parent1 and parent2:
        print(f"  - Parent 1 (Niche {parent1.niche_classes}, Fitness: {parent1.fitness:.2f}%)")
        print(f"  - Parent 2 (Niche {parent2.niche_classes}, Fitness: {parent2.fitness:.2f}%)")
        return parent1, parent2
    else:
        print("  - Could not find a suitable pair of parents from complementary niches. Skipping mating for this generation.")
        return None, None

def merge(parent1, parent2):
    """
    Merges two parent models to create a new child model by averaging their weights.
    This is a simplified version of the paper's "crossover".
    """
    print("Merging parent models to create child...")
    child_wrapper = ModelWrapper(niche_classes=list(range(10)), device=parent1.device)
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

def create_next_generation(current_population, new_child, population_size):
    """
    Creates the next generation's population by combining the old population
    with the new child, then selecting the fittest individuals (elitism).
    """
    print("Creating the next generation...")
    # Evaluate the new child to make sure its fitness is calculated
    evaluate(new_child)

    # Combine the old population with the new child
    full_pool = current_population + [new_child]

    # Sort the entire pool by fitness in descending order
    full_pool.sort(key=lambda x: x.fitness, reverse=True)

    # The next generation consists of the top 'population_size' individuals
    next_generation = full_pool[:population_size]

    print(f"Selected {len(next_generation)} fittest individuals for the next generation.")

    return next_generation

def finetune(model_wrapper, epochs=1):
    """
    Fine-tunes a model on the full dataset. This is useful for the child
    model to learn how to integrate the knowledge from its parents.
    """
    print(f"Fine-tuning model on the full dataset for {epochs} epoch(s)...")
    # Get the dataloader for the full dataset
    train_loader, _ = get_cifar10_dataloaders()
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
