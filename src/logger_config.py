import logging
import sys

def setup_logger(log_file=None):
    """Configures the main logger for the application.

    This function sets up a logger that can write to both the console and
    an optional log file.

    Args:
        log_file (str, optional): The path to the log file. If None,
            file logging is disabled. Defaults to None.
    """
    # Get the root logger
    logger = logging.getLogger("M2N2_SIMULATOR")
    logger.setLevel(logging.DEBUG)

    # Prevent handlers from being added multiple times if the function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a console handler and set the level to info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file path is provided, create a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log each time
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)