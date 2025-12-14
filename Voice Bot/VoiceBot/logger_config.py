import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime('%H:%M:%S,%f')[:-3]

    def format(self, record):
        # Full filename with .py
        record.filename_with_ext = os.path.basename(record.pathname)

        # Build [Process/Thread] label
        parts = []
        if record.processName and record.processName != "MainProcess":
            parts.append(record.processName)
        if record.threadName and record.threadName != "MainThread":
            parts.append(record.threadName)
        
        # Fallback if nothing meaningful
        if not parts:
            parts.append(record.threadName or "Unknown")

        record.proc_thread = "/".join(parts)

        return super().format(record)

def setup_logger(debug_mode: bool = True, log_name: str = "VOICE", log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    delete_old_logs(log_dir, days=3)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    #logger.setLevel(logging.DEBUG if debug_mode else logging.WARNING)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()
    handler_level = logging.DEBUG if debug_mode else logging.WARNING

    formatter = MillisecondFormatter(
        fmt='%(asctime)s %(filename_with_ext)s [%(proc_thread)s] - %(levelname)s :%(lineno)d - %(message)s',
        datefmt='%H:%M:%S,%f'
    )
    log_file = os.path.join(log_dir, f"{log_name}.log")  # base filename

    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",              # Rotate at midnight
        interval=1,
        backupCount=7,                # Keep logs for 7 days
        encoding='utf-8',
        delay=False,
        utc=False                     # Local time
    )

    file_handler.setLevel(handler_level)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"  # Log files will look like: ANPR.log.2025-08-04

    console_handler = logging.StreamHandler()
    console_handler.setLevel(handler_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def delete_old_logs(log_dir: str, days: int = 3):
    now = datetime.now()
    cutoff = now - timedelta(days=days)

    for file in Path(log_dir).glob("*.log.*"):  # Match rotated logs like ANPR.log.2025-08-04
        try:
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            if file_time < cutoff:
                file.unlink()
                print(f"Deleted old log file: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")
