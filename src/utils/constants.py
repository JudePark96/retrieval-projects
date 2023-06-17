import sys

import loguru
from tqdm import tqdm

LOGGING_FORMAT = (
    "<green>{time:YYYY-MM-DDTHH:mm:ss.SSSZ}</green> | "
    "<level>{level: >8}</level> | "
    "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "
    "<cyan>{module}</cyan> | "
    "<cyan>{function}</cyan> |> "
    "<level>{message}</level> "
)

LOGGER = loguru.logger
LOGGER.remove()
LOGGER.add(sink=sys.stdout, level='TRACE', format=LOGGING_FORMAT, enqueue=True, diagnose=True)
LOGGER.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), format=LOGGING_FORMAT, colorize=True)])

