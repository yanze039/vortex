import sys 
import rich
import logging 
from rich.logging import RichHandler

# TODO: reset file handler
# TODO: rich formatting
def initialize_vortex_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler('activations_debug.log')
    file_handler.setLevel(level)
    logging.basicConfig(
        level=level,
        handlers=[RichHandler(rich_tracebacks=True, level=level), file_handler],
        format='%(name)s - %(levelname)s - %(message)s'
    )
    return logger

activations_logger = initialize_vortex_logger('activations_logger', level=logging.DEBUG)