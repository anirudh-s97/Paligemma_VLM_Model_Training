import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


logger.debug("This is a debug message")
logger.info("This is a info message")
logger.error("This is an error message")
logger.warning("This is a warning message")
logger.critical("This is a critical message")