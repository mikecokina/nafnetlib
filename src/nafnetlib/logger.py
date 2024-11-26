import logging
from logging import config as log_conf


logger_config_schema = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(process)d - %(name)s - %(levelname)s: %(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
    },

    "loggers": {
        "main": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": 0
        },
    },

    "root": {
        "level": "DEBUG",
        "handlers": ["console"]
    }
}


def set_up_logging():
    log_conf.dictConfig(logger_config_schema)


# noinspection PyPep8Naming
def getLogger(name, suppress=False):
    if suppress:
        return Logger(name)
    return logging.getLogger(name=name) if not suppress else Logger(name)


class CustomLogger(object):
    _instance = None

    LOGGER = getLogger('nafnetlib', suppress=False)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomLogger, cls).__new__(cls)
            # Put any initialization here.
            set_up_logging()
        return cls._instance

    def disable(self):
        self._instance.LOGGER = getLogger('nafnetlib', suppress=True)

    def enable(self):
        self._instance.LOGGER = getLogger('nafnetlib', suppress=False)

    def get(self):
        return self._instance

    @property
    def logger(self):
        return self.LOGGER

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        return self.logger.exception(*args, **kwargs)

    def debug(self, *args, **kwargs):
        return self.logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.logger.warning(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self.logger.warn(*args, **kwargs)


# noinspection PyPep8Naming
def getPersistentLogger(name):
    return logging.getLogger(name=name)


class Logger(object):
    # noinspection PyUnusedLocal
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


logger = CustomLogger()
