"""
app/logger.py
Logger setup for Agent project
- Colored console output via colorama
- Detailed file logging with full tracebacks
- Per-user log files
"""

import logging
from datetime import datetime
from pathlib import Path
from utils.config import LOG_DIR
from colorama import Fore, Style, init


init(autoreset=True)


def setup_logger(app_name: str) -> logging.Logger:
    """
    Create and configure a per-user logger.

    - Console: colored, INFO level
    - File: detailed with tracebacks, DEBUG level
    """
    logger = logging.getLogger(f"{app_name}")
    logger.setLevel(logging.DEBUG)  # capture everything in file

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # ── Console handler ────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Use formatter with colorama codes (no monkey-patch needed)
    console_formatter = logging.Formatter(
        f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} "
        f"[{Fore.GREEN}%(levelname)s{Style.RESET_ALL}] "
        f"%(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # ── File handler ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file: Path = LOG_DIR / f"{app_name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\n"
        "%(exc_info)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # ── Add handlers and finalize ──────────────────────────────────────
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info("Logger initialized → %s", log_file)
    return logger

LOGGER = setup_logger(app_name='InvestmentApp') 