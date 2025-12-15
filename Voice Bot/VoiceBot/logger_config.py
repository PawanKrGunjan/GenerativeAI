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
        record.filename_with_ext = os.path.basename(record.pathname)
        
        parts = []
        if record.processName and record.processName != "MainProcess":
            parts.append(record.processName)
        if record.threadName and record.threadName != "MainThread":
            parts.append(record.threadName)
        if not parts:
            parts.append(record.threadName or "Unknown")
        
        record.proc_thread = "/".join(parts)
        return super().format(record)


def setup_logger(
    debug_mode: bool = True, 
    log_name: str = "Voicebot", 
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup structured logging for FastAPI Voice Assistant.
    Logs to BOTH file and terminal simultaneously.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    delete_old_logs(log_dir, days=7)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler_level = logging.DEBUG if debug_mode else logging.INFO

    # Formatter for both file and console
    formatter = MillisecondFormatter(
        fmt='%(asctime)s %(filename_with_ext)s [%(proc_thread)s] - %(levelname)-7s :%(lineno)03d - %(message)s'
    )
    
    log_file = os.path.join(log_dir, f"{log_name}.log")

    # ✅ FILE HANDLER - Daily rotation
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding='utf-8',
        delay=False,
        utc=False
    )
    file_handler.setLevel(handler_level)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"

    # ✅ CONSOLE HANDLER - Terminal output (FORCE SHOW)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Always DEBUG for terminal
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"🚀 {log_name} logger initialized → FILE+TERMINAL (debug={debug_mode})")
    return logger


def delete_old_logs(log_dir: str, days: int = 7):
    """Delete log files older than specified days."""
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    deleted_count = 0
    
    for file in Path(log_dir).glob("*.log.*"):
        try:
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            if file_time < cutoff:
                file.unlink()
                deleted_count += 1
        except Exception:
            pass  # Silent cleanup
    
    if deleted_count > 0:
        print(f"🧹 Cleaned {deleted_count} old log files from {log_dir}")


# ✅ REMOVE THIS for FastAPI import
# if __name__ == "__main__":
#     logger = setup_logger(debug_mode=True)
#     logger.info("Test")
