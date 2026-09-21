import logging
import sys


def setup_logger(name="lewm", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger