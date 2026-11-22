"""Log files of an experiment: ``experiment.log`` and a ``run.log`` per run.

While an :class:`~iohblade.experiment.Experiment` runs, every log record
(from BLADE and from libraries) is written to ``<experiment>/experiment.log``.
Records emitted by the thread that executes a run are also written to that
run's ``run.log``. Python warnings are captured into the logs too.

With ``show_stdout=False`` (the default), log handlers that print to the
terminal are detached for the duration of the experiment, so library noise
does not disturb the run overview; it is kept in the files instead. (Older
versions disabled logging entirely, which also lost the messages.)
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

log = logging.getLogger("iohblade")


class RunLogRouter(logging.Handler):
    """Writes records to the ``run.log`` of the run executed by their thread."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self._files: dict[int, str] = {}
        self._lock_files = threading.Lock()
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def register(self, path: str) -> None:
        """Route records of the current thread to ``path``."""
        with self._lock_files:
            self._files[threading.get_ident()] = path

    def unregister(self) -> None:
        """Stop routing records of the current thread."""
        with self._lock_files:
            self._files.pop(threading.get_ident(), None)

    def emit(self, record: logging.LogRecord) -> None:
        path = self._files.get(record.thread)
        if path is None:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(self.format(record) + "\n")
        except Exception:  # never let logging break a run
            self.handleError(record)


_TERMINAL_STREAMS = ("<stdout>", "<stderr>")


def _is_terminal_handler(handler: logging.Handler) -> bool:
    if isinstance(handler, logging.FileHandler):
        return False
    stream = getattr(handler, "stream", None)
    if stream is None:
        return False
    return stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__) or (
        getattr(stream, "name", None) in _TERMINAL_STREAMS
    )


def _all_loggers():
    yield logging.getLogger()
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            yield logger


@contextlib.contextmanager
def experiment_logging(directory: str | None, level: str | int = "INFO", quiet=True):
    """Set up the log files of an experiment for the duration of the block.

    Args:
        directory: The experiment directory (``experiment.log`` goes here);
            None to only route run logs.
        level: Minimum level written to the files.
        quiet: Detach handlers that print to the terminal while running.

    Yields:
        RunLogRouter: Register a run's ``run.log`` with it from the run's
        thread.
    """
    level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {level!r}")
    root = logging.getLogger()
    router = RunLogRouter(level)
    handlers: list[logging.Handler] = [router]
    if directory:
        os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(directory, "experiment.log"), encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(file_handler)

    detached: list[tuple[logging.Logger, logging.Handler]] = []
    if quiet:
        for logger in _all_loggers():
            for handler in list(logger.handlers):
                if _is_terminal_handler(handler):
                    logger.removeHandler(handler)
                    detached.append((logger, handler))

    old_root_level = root.level
    old_iohblade_level = log.level
    old_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)  # undo a disable() from older code
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(min(level, old_root_level) if old_root_level else level)
    log.setLevel(level)
    logging.captureWarnings(True)
    try:
        yield router
    finally:
        logging.captureWarnings(False)
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(old_root_level)
        log.setLevel(old_iohblade_level)
        for logger, handler in detached:
            logger.addHandler(handler)
        logging.disable(old_disable)
