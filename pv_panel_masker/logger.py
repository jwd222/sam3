import logging
import sys
from pathlib import Path

def setup_logger(name: str = "solar_masker", log_file: str = "process.log", level: int = logging.INFO):
    """
    Sets up a universal logger.
    - Console: Shows INFO and above (clean output).
    - File: Shows DEBUG and above (detailed tracing).
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything at the root level

    # Prevent duplicate handlers if script is run multiple times in a notebook/session
    if logger.handlers:
        return logger

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s') # Clean format for console

    # 1. File Handler (Debug + Info + Error)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # 2. Console Handler (Info + Error only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger