# log_setup.py
import logging
from logging.handlers import RotatingFileHandler
from autogen_core import TRACE_LOGGER_NAME, EVENT_LOGGER_NAME

from pathlib import Path
import sys
from typing import Optional


def setup_logging(
    logs_dir: str = "logs",
    log_file: str = "agent.log",
    level: int = logging.INFO,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
    *,
    reconfigure: bool = False,
    console_level: Optional[int] = None,
    autogen_event_level: int = logging.WARNING,
    autogen_trace_level: int = logging.WARNING,
) -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(logs_dir) / log_file

    logger = logging.getLogger("developer_agent")
    logger.setLevel(level)
    logger.propagate = False

    if reconfigure and logger.handlers:
        for h in list(logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(console_level if console_level is not None else level)
        logger.addHandler(ch)

    # Redirect warnings to logging
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(level)
    warnings_logger.propagate = False
    for h in logger.handlers:
        if h not in warnings_logger.handlers:
            warnings_logger.addHandler(h)

    # AutoGen event/trace: file-only (avoid console spam)
    TRACE_LOGGER_NAME = "autogen_core.trace"
    EVENT_LOGGER_NAME = "autogen_core.events"

    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    trace_logger.setLevel(autogen_trace_level)
    event_logger.setLevel(autogen_event_level)
    trace_logger.propagate = False
    event_logger.propagate = False

    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler):
            if h not in trace_logger.handlers:
                trace_logger.addHandler(h)
            if h not in event_logger.handlers:
                event_logger.addHandler(h)

    logger.info("Logging started | file=%s", log_path)
    logger.info(
        "AutoGen EVENT/TRACE logs -> file only | event_level=%s trace_level=%s",
        logging.getLevelName(autogen_event_level),
        logging.getLevelName(autogen_trace_level),
    )
    return logger
