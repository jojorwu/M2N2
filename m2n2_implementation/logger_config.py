import logging
import sys

def setup_logger():
    # Create a logger
    logger = logging.getLogger("M2N2_SIMULATOR")
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the level to debug
    file_handler = logging.FileHandler("simulation.log")
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler and set the level to info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Create a logger instance
logger = setup_logger()