import logging
import sys
from datetime import datetime
from pathlib import Path

# Instead of print(msg)
# logging.info(msg)

# Other logging levels
# logging.debug("Low-level details")
# logging.warning("Something might be wrong")
# logging.error("Something failed")
# logging.critical("Serious failure")

# Idea for exceptions
# try:
#     1 / 0
# except Exception:
#     logging.exception("An error occurred")

# Setup logging
def setup_logging():
    # Create a unique filename per run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_filename = f"./logs/{timestamp}.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create ./logs/ folder if doesn't exist
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)  # ✅ creates folder if missing

    log_filename = log_dir / f"{timestamp}.log"


    # File handler (writes to file)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add both handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Capture print() by redirecting standard out into a logging INFO message
    class PrintToLogger:
        def write(self, message):
            if message.strip():
                logging.info(message.strip())
        def flush(self):
            pass

    sys.stdout = PrintToLogger()