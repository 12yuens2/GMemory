import logging
import os


def get_file_logger(
    name: str,
    log_path: str,
    level: int = logging.INFO,
    echo_console: bool = False,
    fmt: str = '%(asctime)s - %(message)s',
    datefmt: str = '%Y-%m-%d %H:%M:%S',
) -> logging.Logger:
    """
    Return a logger dedicated to `log_path`.

    `logging.getLogger(name)` caches loggers globally by name for the life of
    the process, and handlers are additive - nothing about normal usage ever
    removes a logger's old handlers. In a long-lived process that builds many
    of these loggers over time (e.g. a reused multiprocessing worker running
    one experiment after another), reusing a name leaks the previous
    handler's file descriptor and causes every future log line to also be
    written into the previous handler's file. This resets the named logger's
    handlers before attaching new ones, and disables propagation so messages
    never also land on the root logger's handlers.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    if echo_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

    return logger


CONSOLE_HANDLER_NAME = 'mas-console'


def log_mas_to_console(level: int = logging.DEBUG) -> None:
    """Put the mas package's own logging on stderr, at `level`.

    Only the mas loggers: at DEBUG the root logger would also carry httpx's
    record of every request. Asking twice is asking once - handlers are
    additive, and a second one writes every record a second time.
    """
    logger = logging.getLogger('mas')
    logger.setLevel(level)

    if any(handler.name == CONSOLE_HANDLER_NAME for handler in logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.name = CONSOLE_HANDLER_NAME
    handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
    logger.addHandler(handler)
