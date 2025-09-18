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
    # Get the dataloader for the model's specific niche, using a subset for speed
    train_loader, _ = get_cifar10_dataloaders(niche_classes=model_wrapper.niche_classes, subset_percentage=0.1)
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
    _, test_loader = get_cifar10_dataloaders(subset_percentage=0.1)
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

def evaluate_by_class(model_wrapper):
    """
    Evaluates a model's accuracy on each class of the CIFAR-10 test set.
    Returns a list of accuracies for each class.
    """
    _, test_loader = get_cifar10_dataloaders(subset_percentage=0.1)
    model_wrapper.model.eval()

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model_wrapper.device), target.to(model_wrapper.device)
            output = model_wrapper.model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_accuracies = []
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)

    return class_accuracies

def select_mates(population):
    """
    Selects a complementary pair of parents using an advanced strategy.
    Parent 1: The model with the highest overall fitness.
    Parent 2: The specialist model for Parent 1's weakest class.
    """
    print("Selecting mates with advanced strategy...")
    if not population:
        return None, None

    # 1. Find the best overall model in the population to be Parent 1.
    parent1 = max(population, key=lambda m: m.fitness)
    print(f"  - Parent 1 is the population's best model (Fitness: {parent1.fitness:.2f}%)")

    # 2. Analyze Parent 1 to find its weakest class.
    print("  - Analyzing Parent 1's performance by class...")
    class_accuracies = evaluate_by_class(parent1)
    weakest_class_index = class_accuracies.index(min(class_accuracies))
    print(f"  - Parent 1's weakest class is {weakest_class_index} (Accuracy: {class_accuracies[weakest_class_index]:.2f}%)")

    # 3. Find the specialist for that weakest class to be Parent 2.
    parent2 = None
    # Ensure Parent 2 is not the same model as Parent 1.
    specialist_candidates = [
        m for m in population if m.niche_classes == [weakest_class_index] and m is not parent1
    ]

    if specialist_candidates:
        # From the candidates, pick the one with the highest fitness.
        parent2 = max(specialist_candidates, key=lambda m: m.fitness)
        print(f"  - Found specialist for class {weakest_class_index} as Parent 2 (Fitness: {parent2.fitness:.2f}%)")
    else:
        # Fallback: if no suitable specialist is found, pick the second-best model overall.
        print("  - No suitable specialist found. Using second-best model as fallback Parent 2.")
        sorted_population = sorted(population, key=lambda m: m.fitness, reverse=True)
        if len(sorted_population) > 1:
            parent2 = sorted_population[1]
        else:
            print("  - Not enough models in population to select a second parent.")
            return parent1, None

    if parent1 and parent2:
        return parent1, parent2
    else:
        # This case should be rare given the fallbacks, but is here for safety.
        print("  - Could not select a pair of parents.")
        return None, None

def merge(parent1, parent2, strategy='average'):
    """
    Merges two parent models to create a new child model.
    Supports different merging strategies:
    - 'average': Simple weight averaging.
    - 'fitness_weighted': Weighted average based on parent fitness.
    """
    print(f"Merging parent models to create child using '{strategy}' strategy...")
    child_wrapper = ModelWrapper(niche_classes=list(range(10)), device=parent1.device)
    child_model_state_dict = child_wrapper.model.state_dict()

    parent1_state_dict = parent1.model.state_dict()
    parent2_state_dict = parent2.model.state_dict()

    if strategy == 'fitness_weighted':
        total_fitness = parent1.fitness + parent2.fitness
        # Avoid division by zero if both parents have zero fitness
        if total_fitness == 0:
            weight1, weight2 = 0.5, 0.5
            print("  - Both parents have 0 fitness, falling back to simple average.")
        else:
            weight1 = parent1.fitness / total_fitness
            weight2 = parent2.fitness / total_fitness
            print(f"  - Parent 1 Fitness: {parent1.fitness:.2f}% (Weight: {weight1:.2f})")
            print(f"  - Parent 2 Fitness: {parent2.fitness:.2f}% (Weight: {weight2:.2f})")

        # Apply weighted average
        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] * weight1) + (parent2_state_dict[key] * weight2)

    elif strategy == 'average':
        # Apply simple average
        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] + parent2_state_dict[key]) / 2.0

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

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
    # Get the dataloader for the full dataset, using a subset for speed
    train_loader, _ = get_cifar10_dataloaders(subset_percentage=0.1)
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
