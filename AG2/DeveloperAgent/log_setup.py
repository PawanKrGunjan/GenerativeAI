
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(
    logs_dir: str = "logs",
    log_file: str = "agent.log",
    level: int = logging.INFO,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(logs_dir) / log_file

    logger = logging.getLogger("developer_agent")
    logger.setLevel(level)

    # prevent duplicate handlers if you re-run in same interpreter
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
        logger.addHandler(fh)  # RotatingFileHandler rotates logs by size [web:147]

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Send warnings (like "Function is being overridden") into logging
    logging.captureWarnings(True)  # redirects warnings to the logging system [web:164]

    # Optional: also wire up AutoGen's own loggers (trace/event)
    try:
        from autogen_core import TRACE_LOGGER_NAME, EVENT_LOGGER_NAME
        trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
        event_logger = logging.getLogger(EVENT_LOGGER_NAME)
        trace_logger.setLevel(logging.DEBUG)
        event_logger.setLevel(logging.INFO)
        # Attach same handlers so everything goes to console + file
        for h in logger.handlers:
            trace_logger.addHandler(h)
            event_logger.addHandler(h)
    except Exception:
        pass  # fine if autogen_core isn't available in your install

    logger.info("Logging started | file=%s", log_path)
    return logger


# if __name__ == "__main__":
#     log = setup_logging()