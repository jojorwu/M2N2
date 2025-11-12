"""
Configures the logging setup for the M2N2 simulation.

This module provides a centralized function to configure the application's
logger, ensuring consistent and informative log output.
"""
import logging
import sys

def setup_logger(log_file: str = None):
    """
    Configures the main logger for the M2N2 simulator.

    This function initializes the logger named "M2N2_SIMULATOR". It sets up
    handlers to direct log messages to the console (for INFO level and above)
    and optionally to a file (for DEBUG level and above). The setup is idempotent;
    calling it multiple times will clear existing handlers to avoid duplication.

    Args:
        log_file (str, optional): The path to the log file. If provided,
            logs will be written to this file, overwriting its previous
            content. If None, file logging is disabled. Defaults to None.
    """
    logger = logging.getLogger("M2N2_SIMULATOR")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate logs if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler for INFO messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler for DEBUG messages
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
