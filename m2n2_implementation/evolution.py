import torch
import torch.optim as optim
import torch.nn.functional as F
from model import CifarCNN
from data import get_cifar10_dataloaders
import copy

class ModelWrapper:
    """A wrapper for a neural network model to hold its metadata.

    This class encapsulates a model and its associated evolutionary context,
    such as its specialized niche and fitness score.

    Attributes:
        niche_classes (list[int]): The list of class indices this model is
            specialized in.
        device (str): The device ('cpu' or 'cuda') on which the model's
            tensors are allocated.
        model (CifarCNN): The underlying neural network model instance.
        fitness (float): The fitness score of the model, typically its
            accuracy on a general test set. Initialized to 0.0.
    """
    def __init__(self, niche_classes, device='cpu'):
        self.niche_classes = niche_classes
        self.device = device
        self.model = CifarCNN().to(device)
        # Fitness is measured as accuracy on the full test set.
        self.fitness = 0.0

def specialize(model_wrapper, epochs=1):
    """Trains a model on its specialized data niche.

    This simulates the "resource competition" phase where a model becomes an
    expert in a specific area by training only on data from its assigned niche.

    Args:
        model_wrapper (ModelWrapper): The model wrapper containing the model
            and its niche definition.
        epochs (int, optional): The number of training epochs. Defaults to 1.
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
    """Evaluates the model's fitness on the full test dataset.

    Fitness is defined as the model's accuracy on the general, complete
    test set. This score is stored in the model_wrapper's `fitness` attribute.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.

    Returns:
        float: The calculated accuracy (fitness) of the model as a percentage.
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
    """Evaluates a model's accuracy on each individual class.

    This function is used to identify a model's strengths and weaknesses,
    which is crucial for the advanced mate selection strategy.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.

    Returns:
        list[float]: A list of accuracy percentages, where the index of the
            list corresponds to the class index (0-9 for CIFAR-10).
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
    """Selects a complementary pair of parents using an advanced strategy.

    The strategy is as follows:
    1. Parent 1 is chosen as the model with the highest overall fitness.
    2. Parent 1 is analyzed to find its weakest class.
    3. Parent 2 is chosen as the specialist model for that weakest class,
       ensuring it is a different model from Parent 1.

    This promotes merging models that can compensate for each other's flaws.

    Args:
        population (list[ModelWrapper]): The current population of models.

    Returns:
        tuple[ModelWrapper | None, ModelWrapper | None]: A tuple containing
            the two selected parents. Returns (None, None) if a pair cannot
            be selected.
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
    """Merges two parent models to create a new child model.

    This function combines the weights of two parent models to produce a
    new "child" model.

    Args:
        parent1 (ModelWrapper): The first parent model.
        parent2 (ModelWrapper): The second parent model.
        strategy (str, optional): The merging strategy to use.
            - 'average': Simple arithmetic mean of the weights.
            - 'fitness_weighted': A weighted average where the contribution
              of each parent is proportional to its fitness.
            Defaults to 'average'.

    Returns:
        ModelWrapper: A new model wrapper containing the merged child model.
            The child is initialized as a generalist (all classes).

    Raises:
        ValueError: If an unknown merge strategy is specified.
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
    """Applies random mutations to the model's weights.

    This function introduces genetic diversity into the population by
    randomly altering a fraction of the model's weights in its convolutional
    and linear layers.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to mutate.
        mutation_rate (float, optional): The probability (0.0 to 1.0) that
            any given weight will be mutated. Defaults to 0.01.
        mutation_strength (float, optional): The standard deviation of the
            normal distribution from which mutations are drawn. Controls the
            magnitude of changes. Defaults to 0.1.

    Returns:
        ModelWrapper: The mutated model wrapper.
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
    """Creates the next generation's population via elitism.

    This function combines the existing population with the new child,
    evaluates the child's fitness, and then selects the top individuals
    to form the next generation.

    Args:
        current_population (list[ModelWrapper]): The list of models in the
            current generation.
        new_child (ModelWrapper): The newly created child model.
        population_size (int): The maximum size of the population.

    Returns:
        list[ModelWrapper]: The list of models selected for the next
            generation, sorted by fitness.
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
    """Fine-tunes a model on the full dataset.

    This step is crucial for a newly merged child model, as it helps the
    model learn how to integrate the knowledge from its two parents into a
    cohesive whole.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to fine-tune.
        epochs (int, optional): The number of fine-tuning epochs.
            Defaults to 1.
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
