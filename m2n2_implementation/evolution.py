"""Implements the core evolutionary algorithm for the M2N2 simulation.

This module contains the logic for the main steps of the evolutionary
process, including model specialization, evaluation, mate selection,
merging (crossover), mutation, and generational selection. It operates on
`ModelWrapper` objects, which encapsulate the neural network models and their
associated metadata, conforming to Google's Python docstring style.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import CifarCNN, MnistCNN
from data import get_dataloaders
import copy
import random
from tqdm import tqdm
class ModelWrapper:
    """A wrapper to hold a model and its evolutionary metadata.

    This class encapsulates a model and its associated context, such as its
    specialized niche, its fitness score, and the model architecture.

    Attributes:
        model_name (str): The name of the model architecture ('CIFAR10' or 'MNIST').
        niche_classes (list[int]): A list of class indices the model is
            specialized in. An empty or full list implies a generalist.
        device (str): The device ('cpu' or 'cuda') on which the model's
            tensors are allocated.
        model (torch.nn.Module): The underlying neural network model instance.
        fitness (float): The fitness score of the model, representing its
            accuracy on the full test set. Initialized to 0.0.
    """
    def __init__(self, model_name, niche_classes, device='cpu'):
        """Initializes the ModelWrapper with a model and its niche.

        Args:
            model_name (str): The name of the model to instantiate.
                Supported options: 'CIFAR10', 'MNIST'.
            niche_classes (list[int]): The list of class indices for the
                model's specialized niche.
            device (str, optional): The device to run the model on.
                Defaults to 'cpu'.
        """
        self.model_name = model_name
        self.niche_classes = niche_classes
        self.device = device

        if self.model_name == 'CIFAR10':
            self.model = CifarCNN().to(device)
        elif self.model_name == 'MNIST':
            self.model = MnistCNN().to(device)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Fitness is measured as accuracy on the full test set.
        self.fitness = 0.0

def specialize(model_wrapper, epochs=1):
    """Trains a model in-place on its specialized data niche.

    This simulates the "resource competition" phase where a model becomes an
    expert in a specific area by training only on data from its assigned
    niche. The model's weights are updated directly. This function will
    print its progress to the console.

    Args:
        model_wrapper (ModelWrapper): The model wrapper containing the model
            and its niche definition. Its model will be trained in-place.
        epochs (int, optional): The number of training epochs. Defaults to 1.
    """
    print(f"Specializing model on niche: {model_wrapper.niche_classes} for {epochs} epoch(s)...")
    # Get the dataloader for the model's specific niche, using a subset for speed
    train_loader, _ = get_dataloaders(
        dataset_name=model_wrapper.model_name,
        niche_classes=model_wrapper.niche_classes,
        subset_percentage=0.1
    )
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=0.001)

    model_wrapper.model.train()
    for epoch in range(epochs):
        print(f"  - Epoch {epoch + 1}/{epochs}")
        # Wrap the train_loader with tqdm for a progress bar
        pbar = tqdm(train_loader, desc=f"Specializing Niche {model_wrapper.niche_classes}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(model_wrapper.device), target.to(model_wrapper.device)
            optimizer.zero_grad()
            output = model_wrapper.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
    print("Specialization complete.")

def _get_fitness_score(model_wrapper):
    """Calculates and returns the fitness score for a model.

    This is a lightweight, side-effect-free version of the `evaluate`
    function. It calculates the accuracy on the test set but does *not*
    update the `fitness` attribute of the model wrapper or print any
    output. This makes it suitable for repeated internal use, such as
    during the sequential constructive crossover process.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.

    Returns:
        float: The calculated accuracy (fitness) of the model as a percentage.
    """
    # We always evaluate on the full test set to measure general performance
    _, test_loader = get_dataloaders(dataset_name=model_wrapper.model_name, subset_percentage=0.1)
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
    return accuracy

def evaluate(model_wrapper):
    """Evaluates fitness on the full test set and updates the wrapper.

    Fitness is defined as the model's accuracy on the complete test set.
    As a side effect, this function updates the `fitness` attribute of the
    provided `model_wrapper` object with the calculated score.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate. Its
            `fitness` attribute will be updated.

    Returns:
        float: The calculated accuracy (fitness) of the model as a percentage.
    """
    accuracy = _get_fitness_score(model_wrapper)
    model_wrapper.fitness = accuracy
    return accuracy

def evaluate_by_class(model_wrapper):
    """Evaluates a model's accuracy on each individual class.

    This function is used to identify a model's strengths and weaknesses,
    which is crucial for the advanced mate selection strategy. It does not
    modify the model wrapper.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to evaluate.

    Returns:
        list[float]: A list of 10 accuracy percentages, where the index of
            the list corresponds to a class index from CIFAR-10.
    """
    _, test_loader = get_dataloaders(dataset_name=model_wrapper.model_name, subset_percentage=0.1)
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

    This function promotes "healing" by pairing a strong model with a model
    that is an expert in the first model's weakest area. The strategy is:
    1.  Parent 1 is chosen as the model with the highest overall fitness.
    2.  Parent 1's performance is analyzed to find its weakest class.
    3.  Parent 2 is chosen as the specialist model for that weakest class.
    4.  A fallback is used if a suitable specialist is not found.

    This function prints its selection logic to the console.

    Args:
        population (list[ModelWrapper]): The current population of models.

    Returns:
        tuple[ModelWrapper | None, ModelWrapper | None]: A tuple containing
            the two selected parents. If a suitable pair cannot be found
            (e.g., population is too small), elements can be `None`.
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
    """Merges two parent models into a new child model (crossover).

    This function combines the weights of two parents to produce a new child
    model. The child is initialized as a generalist, with a niche covering
    all classes. This function prints the strategy details to the console.

    Args:
        parent1 (ModelWrapper): The first parent model.
        parent2 (ModelWrapper): The second parent model.
        strategy (str, optional): The merging strategy. Options are:
            - 'average': A simple arithmetic mean of the parent weights.
            - 'fitness_weighted': A weighted average where each parent's
              contribution is proportional to its fitness score.
            - 'layer-wise': Randomly selects each entire layer from one
              of the two parents.
            - 'sequential_constructive': Intelligently builds a child by
              starting with the fitter parent and sequentially swapping in
              layers from the weaker parent if they improve performance.
            Defaults to 'average'.

    Returns:
        ModelWrapper: A new model wrapper containing the merged child model.

    Raises:
        ValueError: If an unknown merge strategy is specified.
    """
    print(f"Merging parent models to create child using '{strategy}' strategy...")
    child_wrapper = ModelWrapper(model_name=parent1.model_name, niche_classes=list(range(10)), device=parent1.device)
    child_model_state_dict = child_wrapper.model.state_dict()

    parent1_state_dict = parent1.model.state_dict()
    parent2_state_dict = parent2.model.state_dict()

    if strategy == 'fitness_weighted':
        total_fitness = parent1.fitness + parent2.fitness
        if total_fitness == 0:
            weight1, weight2 = 0.5, 0.5
        else:
            weight1 = parent1.fitness / total_fitness
            weight2 = parent2.fitness / total_fitness
        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] * weight1) + (parent2_state_dict[key] * weight2)

    elif strategy == 'average':
        for key in child_model_state_dict:
            child_model_state_dict[key] = (parent1_state_dict[key] + parent2_state_dict[key]) / 2.0

    elif strategy == 'layer-wise':
        layer_prefixes = sorted(list(set([k.split('.')[0] for k in parent1_state_dict.keys()])))
        parent_choices = {p: random.choice([1, 2]) for p in layer_prefixes}
        for key in child_model_state_dict:
            prefix = key.split('.')[0]
            child_model_state_dict[key] = parent1_state_dict[key] if parent_choices[prefix] == 1 else parent2_state_dict[key]

    elif strategy == 'sequential_constructive':
        fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        weaker_parent = parent2 if parent1.fitness >= parent2.fitness else parent1
        print(f"  - Using fitter parent (Fitness: {fitter_parent.fitness:.2f}) as base.")

        # Start with the fitter parent's state dict as the best-known state
        best_child_state_dict = copy.deepcopy(fitter_parent.model.state_dict())

        # Evaluate this initial state
        temp_model_wrapper = ModelWrapper(model_name=fitter_parent.model_name, niche_classes=list(range(10)), device=fitter_parent.device)
        temp_model_wrapper.model.load_state_dict(best_child_state_dict)
        best_fitness = _get_fitness_score(temp_model_wrapper)
        print(f"  - Initial child fitness: {best_fitness:.2f}%")

        layer_prefixes = sorted(list(set([k.split('.')[0] for k in fitter_parent.model.state_dict().keys()])))

        for prefix in layer_prefixes:
            # Create a temporary state by swapping in a layer from the weaker parent
            temp_state_dict = copy.deepcopy(best_child_state_dict)
            for key in temp_state_dict:
                if key.startswith(prefix):
                    temp_state_dict[key] = weaker_parent.model.state_dict()[key]

            # Evaluate the temporary state
            temp_model_wrapper.model.load_state_dict(temp_state_dict)
            current_fitness = _get_fitness_score(temp_model_wrapper)

            # If the swap was beneficial, keep it as the new best state
            if current_fitness > best_fitness:
                print(f"  - Swapping layer '{prefix}' improved fitness to {current_fitness:.2f}%. Keeping it.")
                best_fitness = current_fitness
                best_child_state_dict = temp_state_dict
            else:
                print(f"  - Swapping layer '{prefix}' did not improve fitness ({current_fitness:.2f}%). Discarding.")

        # The final child state dict is the one we intelligently constructed
        child_model_state_dict = best_child_state_dict

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    child_wrapper.model.load_state_dict(child_model_state_dict)
    print("Merging complete.")
    return child_wrapper

def mutate(model_wrapper, generation, mutation_rate=0.01, initial_mutation_strength=0.1, decay_factor=0.9):
    """Applies random, adaptively scaled Gaussian mutations to a model's weights.

    This function introduces genetic diversity by altering a fraction of the
    model's weights. The mutation strength is adaptive, decaying
    exponentially with each generation. This allows for larger exploratory
    changes in early generations and smaller, more precise changes later on.
    The mutation is applied in-place.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to mutate.
        generation (int): The current generation number, used to calculate
            the decaying mutation strength.
        mutation_rate (float, optional): The probability that any given
            weight will be chosen for mutation. Defaults to 0.01.
        initial_mutation_strength (float, optional): The initial standard
            deviation for the mutation noise. Defaults to 0.1.
        decay_factor (float, optional): The factor by which the mutation
            strength decays each generation (e.g., 0.9 means 10% decay).
            Defaults to 0.9.

    Returns:
        ModelWrapper: The same model wrapper that was passed in, allowing
            for method chaining.
    """
    # Calculate the decayed mutation strength for the current generation
    decayed_strength = initial_mutation_strength * (decay_factor ** generation)
    print(f"Mutating child model (Gen: {generation}, Strength: {decayed_strength:.4f})...")

    with torch.no_grad():
        for param in model_wrapper.model.parameters():
            if len(param.shape) > 1: # Mutate only multi-dimensional layers (conv, linear)
                # Create a random mask to decide which weights to mutate
                mutation_mask = (torch.rand(param.shape) < mutation_rate).to(model_wrapper.device)
                # Generate random noise scaled by the decayed strength
                mutation = torch.randn(param.shape).to(model_wrapper.device) * decayed_strength
                # Apply the mutation where the mask is True
                param.data += mutation * mutation_mask
    print("Mutation complete.")
    return model_wrapper

def create_next_generation(current_population, new_child, population_size):
    """Creates the next generation's population using elitist selection.

    This function implements the selection step of the algorithm. It combines
    the existing population with the new child, evaluates the child's
    fitness, and then selects the top individuals to form the next
    generation's population. This function prints its progress to the console.

    Args:
        current_population (list[ModelWrapper]): The list of models in the
            current generation.
        new_child (ModelWrapper): The newly created child model to be
            evaluated and included in the selection pool.
        population_size (int): The maximum size of the population.

    Returns:
        list[ModelWrapper]: A new list of models for the next generation,
            sorted by fitness in descending order.
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

def finetune(model_wrapper, epochs=3):
    """Fine-tunes a model in-place on the full dataset with a scheduler.

    This step is crucial for a newly merged child model, as it helps the
    model learn to integrate the knowledge from its two parents into a
    cohesive whole. It uses an Adam optimizer and a learning rate scheduler
    (`StepLR`) that reduces the learning rate over epochs. The model's
    weights are updated directly. This function will print its progress.

    Args:
        model_wrapper (ModelWrapper): The model wrapper to fine-tune. Its
            model will be trained in-place.
        epochs (int, optional): The number of fine-tuning epochs.
            Defaults to 3 to allow the scheduler to take effect.
    """
    print(f"Fine-tuning model for {epochs} epoch(s) with LR scheduler...")
    # Get the dataloader for the full dataset, using a subset for speed
    train_loader, _ = get_dataloaders(dataset_name=model_wrapper.model_name, subset_percentage=0.1)
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    model_wrapper.model.train()
    for epoch in range(epochs):
        print(f"  - Epoch {epoch + 1}/{epochs}")
        pbar = tqdm(train_loader, desc=f"Fine-tuning Child")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(model_wrapper.device), target.to(model_wrapper.device)
            optimizer.zero_grad()
            output = model_wrapper.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

        # Step the scheduler after each epoch
        scheduler.step()

    print("Fine-tuning complete.")