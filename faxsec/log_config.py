"""Shared console logging setup.

Library modules only ever do ``logger = logging.getLogger(__name__)`` and
log through it. Scripts call :func:`setup_logging` once, near the top of
``if __name__ == "__main__":``, to get clean, aligned console output.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


class _ConsoleFormatter(logging.Formatter):
    """Compact console format: ``HH:MM:SS LEVEL  logger.name  message``."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(level: int | str = logging.INFO) -> None:
    """Install a single, formatted console handler on the root logger.

    Idempotent: only the first call installs a handler; later calls just
    adjust the level. Level can be overridden globally via the
    ``FAXSEC_LOG_LEVEL`` environment variable (e.g. ``DEBUG`` for verbose
    per-iteration fit progress).
    """
    global _CONFIGURED

    level = os.environ.get("FAXSEC_LOG_LEVEL", level)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if _CONFIGURED:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(_ConsoleFormatter())
    root_logger.addHandler(handler)

    _CONFIGURED = True
