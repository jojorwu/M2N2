import yaml
from typing import Dict, Any

class ConfigManager:
    """Manages the loading and validation of the simulation configuration."""

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Loads and validates the configuration file.

        Args:
            config_path (str): The path to the configuration YAML file.

        Raises:
            ValueError: If the config file is missing, malformed, or empty.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                if not isinstance(self.config, dict):
                    raise ValueError("Configuration file is empty or invalid.")
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a value from the configuration."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Gets a value from the configuration using dictionary-style access."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets a value in the configuration using dictionary-style access."""
        self.config[key] = value
