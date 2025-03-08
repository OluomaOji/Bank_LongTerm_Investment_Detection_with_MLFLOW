import logging
import os

# set directory
Log_dir = "Logs"
if not os.path.exists(Log_dir):
    os.makedirs(Log_dir)

# Set the Logging Configuration
logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Log_dir,"bank_loan_detector.logs")),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    return logging.getLogger(name)