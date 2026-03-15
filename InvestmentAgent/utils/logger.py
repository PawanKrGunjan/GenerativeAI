"""
app/logger.py
Production-ready logger for Agent system

Features:
• Colored console output (only in debug mode)
• Rotating file logs (daily + size limit)
• Detailed file logs with process/thread info
• Optional JSON structured logging
• Controlled verbosity via mode: "debug" or "production"
• Clean stack traces with full exception info
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

try:
    from utils.config import LOG_DIR
except ImportError:
    LOG_DIR = Path("logs")  # fallback

try:
    from colorama import Fore, Style, init
    COLORAMA_AVAILABLE = True
    init(autoreset=True)
except ImportError:
    COLORAMA_AVAILABLE = False

# ────────────────────────────────────────────────
# Formatter Classes
# ────────────────────────────────────────────────

class ColoredFormatter(logging.Formatter):
    """Colored console output (only used in debug mode)"""
    
    LEVEL_COLORS = {
        "DEBUG":    Fore.BLUE,
        "INFO":     Fore.GREEN,
        "WARNING":  Fore.YELLOW,
        "ERROR":    Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    } if COLORAMA_AVAILABLE else {}

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL if COLORAMA_AVAILABLE else ""

        # Colorize level name
        level_colored = f"{color}{record.levelname:<8}{reset}"
        
        # Short timestamp for console
        asctime = self.formatTime(record, self.datefmt)
        
        return (
            f"{asctime} | {level_colored} | "
            f"{record.filename}:{record.funcName}:{record.lineno} | "
            f"{record.getMessage()}"
        )


class PlainFormatter(logging.Formatter):
    """Clean formatter for production console (no colors)"""
    def format(self, record: logging.LogRecord) -> str:
        return (
            f"{self.formatTime(record, self.datefmt)} | "
            f"{record.levelname:<8} | "
            f"{record.filename}:{record.funcName}:{record.lineno} | "
            f"{record.getMessage()}"
        )


def get_json_formatter() -> logging.Formatter:
    """Structured JSON formatter – very useful in production"""
    try:
        import json
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                data = {
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
                    "level": record.levelname,
                    "logger": record.name,
                    "file": f"{record.filename}:{record.lineno}",
                    "function": record.funcName,
                    "process": record.process,
                    "thread": record.thread,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    data["exception"] = self.formatException(record.exc_info)
                return json.dumps(data, ensure_ascii=False)
        return JsonFormatter()
    except ImportError:
        # fallback to detailed text if json not available
        return logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d | PID:%(process)d | TID:%(thread)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


# ────────────────────────────────────────────────
# Main Logger Setup
# ────────────────────────────────────────────────

def setup_logger(
    app_name: str = "InvestmentAgent",
    mode: Literal["debug", "production"] = "production",
    json_output: bool = False,
    log_dir: Path = LOG_DIR,
) -> logging.Logger:
    """
    Production-ready logger factory
    
    Args:
        app_name:   Name of the application/module
        mode:       "debug" → more verbose console, colors
                    "production" → minimal console, detailed files
        json_output: Whether to output JSON logs to file (good for log aggregation)
        log_dir:    Where to store log files
    """
    logger = logging.getLogger(app_name)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)          # always capture everything
    logger.propagate = False

    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    base_filename = f"{app_name}_{timestamp}.log"
    log_file = log_dir / base_filename

    # ─── Console Handler ───────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    
    if mode == "debug":
        console_handler.setLevel(logging.DEBUG)
        formatter = ColoredFormatter(datefmt="%H:%M:%S")
    else:
        console_handler.setLevel(logging.INFO)
        formatter = PlainFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ─── File Handler (rotating) ───────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,      # 10 MB
        backupCount=7,                  # keep 7 days
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)

    if json_output:
        file_handler.setFormatter(get_json_formatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d | PID:%(process)d | TID:%(thread)d | "
            "%(message)s  |  %(funcName)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f"
        ))
    
    logger.addHandler(file_handler)

    # Initial log
    logger.info(
        f"Logger initialized | mode={mode} | json={json_output} | file={log_file}"
    )

    return logger


# Usage examples:
# ------------------------------------------------------
# Development / debugging
LOGGER = setup_logger("InvestmentAgent", mode="debug")

# Production (Docker, server, etc.)
# LOGGER = setup_logger("InvestmentAgent", mode="production", json_output=True)