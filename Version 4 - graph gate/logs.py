import logging
import sys
from datetime import datetime

# Setup logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    filename=f"./logs/log_{timestamp}.txt", # Need to run the program from Version X directory
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Capture print()
class PrintToLogger:
    def write(self, message):
        if message.strip():
            logging.info(message.strip())
    def flush(self):
        pass

sys.stdout = PrintToLogger()