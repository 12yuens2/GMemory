"""Echoing the mas loggers to stderr, however many times it is asked for.

Every spawned worker installs its own settings, so the call is made once per
experiment rather than once per run.
"""

import logging

import pytest

from mas.logging_utils import log_mas_to_console


@pytest.fixture(autouse=True)
def clean_mas_logger():
    logger = logging.getLogger('mas')
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    for handler in handlers:
        logger.removeHandler(handler)
    yield logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate


def test_the_mas_loggers_reach_stderr(clean_mas_logger, capsys):
    log_mas_to_console()

    logging.getLogger('mas.memory').debug('a memory update prompt')

    assert capsys.readouterr().err.count('a memory update prompt') == 1


def test_asking_twice_does_not_double_every_line(clean_mas_logger, capsys):
    log_mas_to_console()
    log_mas_to_console()

    logging.getLogger('mas.memory').debug('a memory update prompt')

    assert capsys.readouterr().err.count('a memory update prompt') == 1, (
        'a second handler on the same logger writes every record twice'
    )
