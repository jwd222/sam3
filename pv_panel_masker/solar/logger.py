import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "solar_masker", log_dir: str = "logs", level: int = logging.INFO):
    """
    Sets up a universal logger with timestamped log files.
    - Console: Shows INFO and above (clean output).
    - File: Shows DEBUG and above (detailed tracing), saved in log_dir.
    
    Args:
        name: Name of the logger
        log_dir: Directory to save log files
        level: Console logging level
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything at the root level

    # Prevent duplicate handlers if script is run multiple times
    if logger.handlers:
        return logger

    # 1. Prepare Log File Path
    # Create directory if it doesn't exist
    log_folder = Path(log_dir)
    log_folder.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename: process_YYYYMMDD_HHMMSS.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"process_{timestamp}.log"
    log_filepath = log_folder / log_filename

    # 2. Create Formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s') # Clean format for console

    # 3. File Handler (Debug + Info + Error)
    # Using str(log_filepath) for compatibility
    file_handler = logging.FileHandler(str(log_filepath), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # 4. Console Handler (Info + Error only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # 5. Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Optional: Log where the file is being saved for debug purposes
    logger.debug(f"Logger initialized. Saving logs to: {log_filepath.absolute()}")

    return logger