import logging
import os
import sys
from datetime import datetime 
import time
from transformers.utils.logging import log_levels


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'logs',LOG_FILE)

# Creates logs directory and logfile if not present else it appends the information
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Whenever the logging.info() is intiated and adds the logs to the log file
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s %(message)s",
    level= logging.INFO
)

def config_logger(log_level: str):
    _log_level = log_levels[log_level]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=_log_level
    )
    return _log_level