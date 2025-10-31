import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import torch
import yaml
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import EvolutionSimulator
from src.model import CifarCNN
from src.config_manager import ConfigManager
from src.model_factory import create_model

class TestSimulatorInitialization(unittest.TestCase):
    """Unit tests for the EvolutionSimulator's initialization logic."""

    def setUp(self):
        self.test_dir = "tests/temp_simulator_test"
        self.config_path = os.path.join(self.test_dir, "temp_config.yaml")
        os.makedirs(self.test_dir, exist_ok=True)
        self.base_config = {
            'model_name': 'CIFAR10',
            'dataset_name': 'CIFAR10',
            'precision_config': '32',
            'num_generations': 1,
            'population_size': 1,
            'mate_selection_strategy': 'healing',
            'generation_strategy': 'replace_worst',
            'merge_strategy': 'average',
            'fitness_weighted_merge_dampening_factor': 25.0,
            'optimizer_config': {'learning_rate': 0.001},
            'scheduler_config': {'patience': 2, 'factor': 0.5},
            'batch_size': 64,
            'mutation_rate': 0.0,
            'initial_mutation_strength': 0.0,
            'mutation_decay_factor': 1.0,
            'subset_percentage': 0.01,
            'validation_split': 0.1,
            'default_epochs': {'specialize': 0, 'finetune': 0},
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if os.path.exists("command_config.json"):
            os.remove("command_config.json")
        model_dir = "src/pretrained_models"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    def test_save_final_population_preserves_unrelated_files(self):
        model_dir = "src/pretrained_models"
        os.makedirs(model_dir, exist_ok=True)
        loaded_model_path = os.path.join(model_dir, "model_niche_0_fitness_10.00.pth")
        user_file_path = os.path.join(model_dir, "user_backup_model.pth")
        dummy_model = CifarCNN()
        torch.save(dummy_model.state_dict(), loaded_model_path)
        with open(user_file_path, "w") as f:
            f.write("This is a user backup, do not delete.")
        config = self.base_config.copy()
        config['num_generations'] = 1
        config['population_size'] = 1
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.run()
        self.assertTrue(os.path.exists(user_file_path), "The user's unrelated file was deleted.")
        self.assertFalse(os.path.exists(loaded_model_path), "The original loaded model file was not deleted.")
        new_model_files = [f for f in os.listdir(model_dir) if f.startswith('model_niche_')]
        self.assertGreater(len(new_model_files), 0, "No new model was saved to the directory.")

    def test_clear_simulation_artifacts_deletes_log_file(self):
        log_file_path = "fitness_log.csv"
        with open(log_file_path, "w") as f:
            f.write("generation,best_fitness,average_fitness\n")
            f.write("1,10.0,5.0\n")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator._clear_simulation_artifacts()
        self.assertFalse(os.path.exists(log_file_path), "The fitness_log.csv file was not deleted.")

    @patch('src.simulator.glob.glob')
    def test_loaded_model_fitness_is_marked_as_stale(self, mock_glob):
        model_dir = os.path.join(self.test_dir, "pretrained_models")
        os.makedirs(model_dir, exist_ok=True)
        dummy_model_path = os.path.join(model_dir, "model_niche_0_fitness_99.9.pth")
        torch.save(CifarCNN().state_dict(), dummy_model_path)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        mock_glob.return_value = [dummy_model_path]
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertGreater(len(simulator.population), 0, "Simulator failed to load any models.")
        self.assertFalse(simulator.population[0].fitness_is_current)

    @patch('src.simulator.setup_logger')
    def test_logging_is_configurable(self, mock_setup_logger):
        config_with_log = self.base_config.copy()
        log_file_path = os.path.join(self.test_dir, "test.log")
        config_with_log['log_file'] = log_file_path
        with open(self.config_path, 'w') as f:
            yaml.dump(config_with_log, f)
        EvolutionSimulator(config_path=self.config_path)
        mock_setup_logger.assert_called_with(log_file=log_file_path)

        config_without_log = self.base_config.copy()
        config_without_log['log_file'] = None
        with open(self.config_path, 'w') as f:
            yaml.dump(config_without_log, f)
        EvolutionSimulator(config_path=self.config_path)
        mock_setup_logger.assert_called_with(log_file=None)

    def test_simulator_uses_seed_from_config(self):
        config_with_seed = self.base_config.copy()
        expected_seed = 12345
        config_with_seed['seed'] = expected_seed
        with open(self.config_path, 'w') as f:
            yaml.dump(config_with_seed, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.config_manager.seed, expected_seed, "Simulator did not use the seed from the config file.")

    @patch('src.config_manager.np.random.randint', return_value=54321)
    def test_simulator_generates_random_seed_if_not_provided(self, mock_randint):
        config_without_seed = self.base_config.copy()
        if 'seed' in config_without_seed:
            del config_without_seed['seed']
        with open(self.config_path, 'w') as f:
            yaml.dump(config_without_seed, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.config_manager.seed, 54321, "Simulator did not generate a random seed when none was provided.")
        mock_randint.assert_called_once()

    @patch('src.simulator.EvolutionSimulator._run_specialization_phase')
    @patch('src.simulator.EvolutionSimulator._run_evaluation_phase')
    @patch('src.simulator.select_mates', return_value=(None, None))
    def test_simulator_loads_dynamic_config_from_command_file(self, mock_select_mates, mock_evaluation, mock_specialization):
        config = self.base_config.copy()
        config['merge_strategy'] = 'average'
        config['mutation_rate'] = 0.1
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        command_config = {"merge_strategy": "fitness_weighted", "mutation_rate": 0.99}
        with open("command_config.json", 'w') as f:
            json.dump(command_config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.config_manager.merge_strategy, 'average')
        self.assertEqual(simulator.config_manager.mutation_rate, 0.1)
        simulator.run()
        self.assertEqual(simulator.config_manager.merge_strategy, 'fitness_weighted', "Merge strategy was not dynamically updated.")
        self.assertEqual(simulator.config_manager.mutation_rate, 0.99, "Mutation rate was not dynamically updated.")
        mock_select_mates.assert_called_once()

    @patch('src.simulator.specialize')
    @patch('src.simulator.get_dataloaders')
    @patch('src.simulator.glob.glob')
    def test_specialization_is_skipped_for_generalist_with_custom_num_classes(self, mock_glob, mock_get_dataloaders, mock_specialize):
        mock_glob.return_value = []
        custom_num_classes = 5
        mock_get_dataloaders.return_value = (None, None, None, custom_num_classes)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        simulator = EvolutionSimulator(config_path=self.config_path)
        self.assertEqual(simulator.num_classes, custom_num_classes)
        mock_specialize.reset_mock()
        from src.model_wrapper import ModelWrapper
        from src.enums import ModelName
        model = create_model(ModelName.CIFAR10, custom_num_classes, 'cpu')
        generalist_wrapper = ModelWrapper(
            model_name=ModelName.CIFAR10,
            model=model,
            niche_classes=list(range(custom_num_classes)),
            device='cpu'
        )
        simulator.population = [generalist_wrapper]
        simulator._run_specialization_phase(generation=1)
        mock_specialize.assert_not_called()

    def test_sequential_constructive_merge_with_zero_validation_split_raises_error(self):
        self.base_config['merge_strategy'] = 'sequential_constructive'
        self.base_config['validation_split'] = 0.0
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        with self.assertRaises(ValueError) as cm:
            EvolutionSimulator(config_path=self.config_path)
        self.assertIn("The 'sequential_constructive' merge strategy requires a validation_split > 0", str(cm.exception))

    @patch('src.simulator.plot_fitness_history')
    @patch('src.simulator.EvolutionSimulator._save_final_population')
    @patch('src.simulator.EvolutionSimulator._initialize_population')
    def test_summarize_and_save_orchestrates_correctly(self, mock_init_pop, mock_save_population, mock_plot_fitness):
        """
        Tests that _summarize_and_save correctly calls its helper methods
        for plotting and saving.
        """
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.fitness_history = [(10.0, 5.0)]

        from src.model_wrapper import ModelWrapper
        model = create_model(simulator.config_manager.model_name, simulator.num_classes, simulator.device)
        wrapper = ModelWrapper(model_name=simulator.config_manager.model_name, model=model, niche_classes=[0], device='cpu')
        wrapper.fitness = 10.0
        simulator.population = [wrapper]

        simulator._summarize_and_save()

        mock_plot_fitness.assert_called_once_with(simulator.fitness_history, 'fitness_history.png')
        mock_save_population.assert_called_once()

    @patch('src.simulator.EvolutionSimulator._delete_old_models')
    @patch('src.simulator.EvolutionSimulator._initialize_population')
    def test_save_final_population_saves_all_models(self, mock_init_pop, mock_delete_old_models):
        """
        Tests that _save_final_population calls the save method for each
        model in the population.
        """
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        simulator = EvolutionSimulator(config_path=self.config_path)

        # Create a mock population
        mock_model_1 = MagicMock()
        mock_model_1.niche_classes = [0]
        mock_model_1.fitness = 99.88
        mock_model_2 = MagicMock()
        mock_model_2.niche_classes = [1, 2]
        mock_model_2.fitness = 88.99
        simulator.population = [mock_model_1, mock_model_2]

        simulator._save_final_population()

        mock_model_1.save.assert_called_once()
        mock_model_2.save.assert_called_once()
        mock_delete_old_models.assert_called_once()


    @patch('src.simulator.EvolutionSimulator._initialize_population')
    def test_delete_old_models_clears_correct_files(self, mock_initialize_population):
        model_dir = "src/pretrained_models"
        os.makedirs(model_dir, exist_ok=True)
        loaded_model_path = os.path.join(model_dir, "loaded_model.pth")
        sim_generated_path = os.path.join(model_dir, "model_niche_1_fitness_5.0.pth")
        user_backup_path = os.path.join(model_dir, "user_backup.pth")

        for p in [loaded_model_path, sim_generated_path, user_backup_path]:
            with open(p, "w") as f: f.write("dummy content")

        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.loaded_model_files = [loaded_model_path]

        simulator._delete_old_models(model_dir)

        self.assertFalse(os.path.exists(loaded_model_path))
        self.assertFalse(os.path.exists(sim_generated_path))
        self.assertTrue(os.path.exists(user_backup_path))

    @patch('src.simulator.EvolutionSimulator._initialize_population')
    @patch('src.simulator.EvolutionSimulator._delete_old_models')
    def test_save_final_population_skips_delete_when_flag_is_false(self, mock_delete_old_models, mock_initialize_population):
        config = self.base_config.copy()
        config['delete_old_models'] = False
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        simulator = EvolutionSimulator(config_path=self.config_path)

        mock_model = MagicMock()
        mock_model.niche_classes = [0]
        mock_model.fitness = 10.0
        simulator.population = [mock_model]

        simulator._save_final_population()

        mock_delete_old_models.assert_not_called()

    @patch('src.simulator.EvolutionSimulator._initialize_population')
    def test_evaluation_phase_handles_empty_population_gracefully(self, mock_init_pop):
        """
        Tests that _run_evaluation_phase handles an empty population gracefully
        by logging an error and returning, instead of crashing.
        """
        with open(self.config_path, 'w') as f:
            yaml.dump(self.base_config, f)

        simulator = EvolutionSimulator(config_path=self.config_path)
        simulator.population = []

        # Use assertLogs to check for the expected error message
        with self.assertLogs('M2N2_SIMULATOR', level='ERROR') as cm:
            simulator._run_evaluation_phase()
            # Verify that the correct error message was logged
            self.assertIn("Population is empty. Cannot run evaluation.", cm.output[0])

if __name__ == '__main__':
    unittest.main()
