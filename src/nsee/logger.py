import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    console: bool = True,
    log_file: Optional[Path] = None,
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging in case of root logger config
    if logger.handlers:
        return logger  # Prevent duplicate handlers

    # File formatter (no color)
    if log_file:
        file_formatter = logging.Formatter(
            "%(asctime)s [%(module)10s:%(lineno)3d] !%(levelno)2d  %(message)s",
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    # Console formatter
    if console:
        # Define colored level names
        logging.addLevelName(
            logging.INFO, f"\033[92m{logging.getLevelName(logging.INFO)}\033[0m"
        )
        logging.addLevelName(
            logging.WARNING, f"\033[33m{logging.getLevelName(logging.WARNING)}\033[0m"
        )
        logging.addLevelName(
            logging.ERROR, f"\033[31m{logging.getLevelName(logging.ERROR)}\033[0m"
        )
        logging.addLevelName(
            logging.CRITICAL, f"\033[41m{logging.getLevelName(logging.CRITICAL)}\033[0m"
        )
        console_formatter = logging.Formatter(
            "%(asctime)s [%(module)10s:%(lineno)3d] %(levelname)8s  %(message)s",
            datefmt="%H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    logger = setup_logger("report", level=10)
    logger.info("Log started...")
