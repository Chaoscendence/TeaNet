import logging
import os

def get_logger(log_file_name):  

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    if not os.path.exists('./Logs'):
        os.makedirs('./Logs')

    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger