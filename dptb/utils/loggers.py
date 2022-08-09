"""Logger initialization for package."""

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pathlib import Path
logging.getLogger(__name__)

__all__ = ["set_log_handles"]

# logger formater
FFORMATTER = logging.Formatter(
    "[%(asctime)s] %(app_name)s %(levelname)-7s %(name)-45s %(message)s"
)
CFORMATTER = logging.Formatter(
#    "%(app_name)s %(levelname)-7s |-> %(name)-45s %(message)s"
    "%(app_name)s %(levelname)-7s %(message)s"
)

class _AppFilter(logging.Filter):
    """Add field `app_name` to log messages."""

    def filter(self, record):
        record.app_name = "DEEPTB"
        return True


def set_log_handles(
    level: int,
    log_path: Optional["Path"] = None
):
    """Set desired level for package loggers and add file handlers.

    Parameters
    ----------
    level: int
        logging level
    log_path: Optional[str]
        path to log file, if None logs will be send only to console. If the parent
        directory does not exist it will be automatically created, by default None
    mpi_log : Optional[str], optional
        mpi log type. Has three options. `master` will output logs to file and console
        only from rank==0. `collect` will write messages from all ranks to one file
        opened under rank==0 and to console. `workers` will open one log file for each
        worker designated by its rank, console behaviour is the same as for `collect`.
        If this argument is specified, package 'mpi4py' must be already installed.
        by default None

    Raises
    ------
    RuntimeError
        If the argument `mpi_log` is specified, package `mpi4py` is not installed.

    References
    ----------
    https://groups.google.com/g/mpi4py/c/SaNzc8bdj6U
    https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
    https://stackoverflow.com/questions/56085015/suppress-openmp-debug-messages-when-running-tensorflow-on-cpu

    Notes
    -----
    Logging levels:

    +---------+--------------+----------------+----------------+----------------+
    |         | our notation | python logging | tensorflow cpp | OpenMP         |
    +=========+==============+================+================+================+
    | debug   | 10           | 10             | 0              | 1/on/true/yes  |
    +---------+--------------+----------------+----------------+----------------+
    | info    | 20           | 20             | 1              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+
    | warning | 30           | 30             | 2              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+
    | error   | 40           | 40             | 3              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+

    """
    # silence logging for OpenMP when running on CPU if level is any other than debug
    if level <= 10:
        os.environ["KMP_WARNINGS"] = "FALSE"

    # set TF cpp internal logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(int((level / 10) - 1))

    # get root logger
    root_log = logging.getLogger()

    # remove all old handlers
    root_log.setLevel(level)
    for hdlr in root_log.handlers[:]:
        root_log.removeHandler(hdlr)

    # * add console handler ************************************************************
    ch = logging.StreamHandler()
    ch.setFormatter(CFORMATTER)

    ch.setLevel(level)
    ch.addFilter(_AppFilter())
    root_log.addHandler(ch)

    # * add file handler ***************************************************************
    if log_path:

        # create directory
        log_path.parent.mkdir(exist_ok=True, parents=True)

        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(FFORMATTER)

        if fh:
            fh.setLevel(level)
            fh.addFilter(_AppFilter())
            root_log.addHandler(fh)
