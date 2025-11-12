import unittest
from unittest.mock import Mock, patch
from src.selection_strategies import HealingMateSelectionStrategy
from src.model_wrapper import ModelWrapper
from src.enums import ModelName, DatasetName
from src.config_manager import ConfigManager

class TestHealingMateSelectionStrategy(unittest.TestCase):

    def setUp(self):
        """Set up a mock population for testing."""
        self.strategy = HealingMateSelectionStrategy()

        # Create mock models
        self.parent1 = Mock(spec=ModelWrapper)
        self.parent1.fitness = 95.0
        self.parent1.model_name = ModelName.CNN
        self.parent1.niche_classes = list(range(10)) # Generalist

        self.fallback_model = Mock(spec=ModelWrapper)
        self.fallback_model.fitness = 90.0
        self.fallback_model.niche_classes = list(range(10)) # Generalist

        self.specialist_weakest = Mock(spec=ModelWrapper)
        self.specialist_weakest.niche_classes = [0] # Specialist for the weakest class
        self.specialist_weakest.fitness = 80.0

        self.specialist_second_weakest = Mock(spec=ModelWrapper)
        self.specialist_second_weakest.niche_classes = [1] # Specialist for the second weakest class
        self.specialist_second_weakest.fitness = 85.0

        self.population = [
            self.parent1,
            self.fallback_model,
            self.specialist_weakest,
            self.specialist_second_weakest
        ]

    def test_select_mates_chooses_specialist_for_second_weakest_class(self):
        """
        Test that the strategy selects a specialist for the second-weakest class
        if no specialist for the absolute weakest class is available.
        """
        # Mock class accuracies for parent1: Class 0 is weakest, Class 1 is second-weakest.
        # Accuracies: [10.0, 20.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0]
        mock_accuracies = [10.0, 20.0] + [90.0 + i for i in range(8)]
        self.parent1.evaluate_by_class.return_value = mock_accuracies

        # The population for this test case should NOT have a specialist for class 0
        population_without_weakest_specialist = [
            self.parent1,
            self.fallback_model,
            self.specialist_second_weakest
        ]

        # Execute the strategy
        config_manager = Mock(spec=ConfigManager)
        config_manager.dataset_name = DatasetName.CIFAR10
        config_manager.subset_percentage = 1.0
        config_manager.seed = 42
        pairs = self.strategy.select_parent_pairs(
            population_without_weakest_specialist,
            1,
            config_manager=config_manager
        )

        # Assertions
        self.assertEqual(len(pairs), 1)
        selected_parent1, selected_parent2 = pairs[0]
        self.assertIsNotNone(selected_parent2, "A second parent should have been selected.")
        self.assertIs(selected_parent1, self.parent1, "Parent 1 should be the model with the highest fitness.")
        self.assertIs(selected_parent2, self.specialist_second_weakest,
                      "Parent 2 should be the specialist for the second-weakest class (class 1).")
        self.assertNotEqual(selected_parent2, self.fallback_model,
                            "The fallback model should not have been chosen.")

    def test_select_mates_falls_back_when_no_specialist_exists(self):
        """
        Test that the strategy falls back to the second-best model when no
        specialists for any of the top 3 weakest classes are available.
        """
        # Mock class accuracies for parent1: Classes 0, 1, 2 are weakest.
        mock_accuracies = [10.0, 20.0, 30.0] + [90.0 + i for i in range(7)]
        self.parent1.evaluate_by_class.return_value = mock_accuracies

        # Population with no relevant specialists
        population_without_any_specialist = [self.parent1, self.fallback_model]

        # Execute the strategy
        config_manager = Mock(spec=ConfigManager)
        config_manager.dataset_name = DatasetName.CIFAR10
        config_manager.subset_percentage = 1.0
        config_manager.seed = 42
        pairs = self.strategy.select_parent_pairs(
            population_without_any_specialist,
            1,
            config_manager=config_manager
        )

        # Assertions
        self.assertEqual(len(pairs), 1)
        selected_parent1, selected_parent2 = pairs[0]
        self.assertIs(selected_parent1, self.parent1)
        self.assertIs(selected_parent2, self.fallback_model,
                      "Parent 2 should be the fallback model (second-highest fitness).")

if __name__ == '__main__':
    unittest.main()
