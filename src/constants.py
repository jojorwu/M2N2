"""
Shared constants for the M2N2 simulation.

This module centralizes constants used across the application to avoid hardcoding
values in multiple places. Adhering to the DRY (Don't Repeat Yourself) principle
makes the code more maintainable and less prone to errors.
"""

# The official filename for the fitness log CSV.
FITNESS_LOG_FILENAME: str = "fitness_log.csv"

# The official filename for the command file used by the dashboard.
COMMAND_FILE: str = "command.json"

# The directory for caching tokenized LLM datasets to speed up subsequent runs.
LLM_CACHE_DIR: str = "src/cache"
