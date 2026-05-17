import logging

log_format = "%(asctime)s | %(levelname)-5s %(funcName)-30s %(message)s"
log_level = logging.DEBUG
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())