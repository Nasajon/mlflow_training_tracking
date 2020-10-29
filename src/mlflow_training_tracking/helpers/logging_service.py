import os
import logging
import pathlib
from logging.handlers import RotatingFileHandler


class LoggingService:
    #__logger = None

    def __init__(self, logname, filename=None):
        #if LoggingService.__logger is not None:
        #    raise RuntimeError("LoggingService already initialized")
        self.filename = filename
        self.logname = logname
        self.__logger = None

    def get_logger(self):
        if self.__logger is None:
            self.__logger = logging.getLogger(self.logname)
            self.__logger.setLevel(level=logging.DEBUG)
            for handler in self._get_handlers():
                self.__logger.addHandler(handler)
        return self.__logger

    def _get_handlers(self):
        handlers = set()
        log_rotate_folder = str(pathlib.Path.home().joinpath(
            '.logs/mlflow_training_tracking'))
        log_format = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(log_format)
        handlers.add(stream_handler)

        if not os.path.exists(log_rotate_folder):
            os.makedirs(log_rotate_folder)

        file_handler = RotatingFileHandler(
            f'{log_rotate_folder}/runlogs', maxBytes=20 * 2**20, backupCount=2)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        handlers.add(file_handler)
        if self.filename:
            file_handler = logging.FileHandler(self.filename, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_format)
            handlers.add(file_handler)

        return handlers


# loggingService = LoggingService("runid", filename='runid.log')
# log = loggingService.get_logger()
# log.debug("teste debug")
# log.info("teste info")
# # import time
# # time.sleep(0.2)
# for i in range(2**20):
#     log.error(f"teste erro {i} of {2**20}")
