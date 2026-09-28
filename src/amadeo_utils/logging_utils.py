"""
Logging to a file as well as the screen, for the long-running servers.

Every server in this tree logs to the console through logging.basicConfig. add_log_file() adds a second destination
to the same (root) logger, so every line that reaches the screen also reaches the file: a server run in tmux keeps
its pane, and the file survives the pane, can be read by the rest of the group, and can be searched later.

Each server reads 'log_file' from its own JSON config and calls add_log_file() with it once its settings are loaded.
No 'log_file' (or an empty one) means screen only - exactly the behaviour before this existed.

The file:
  * starts a new file at midnight; the old one is renamed <file>.YYYY-MM-DD, and the last LOG_RETENTION_DAYS of them
    are kept (older ones are deleted at the next rollover);
  * uses the same line format as the console, without the colour codes (ColoredText's ANSI escapes would otherwise
    show up as '^[[1;34m' junk in the file);
  * is readable by its owner and group only - server logs can hold what was said near a microphone.
"""

import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

# The console format every server in this tree already uses (see their logging.basicConfig calls)
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'

# How many daily files are kept
LOG_RETENTION_DAYS = 90

# ANSI colour / style escapes, e.g. '\033[1;34m' ... '\033[0m'
_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


class PlainFormatter(logging.Formatter):
    """A Formatter that strips ANSI colour codes, so a log file reads as plain text."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a record as the base class does, then removes any colour codes from the result.

        :param record: The log record.
        :return: The formatted line, without ANSI escapes.
        """
        return _ANSI_ESCAPE.sub('', super().format(record))


class PrivateTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    A TimedRotatingFileHandler whose files are readable by their owner and group only (0660): the current file, and
    each new one it opens after a rollover (the rotated copies are renames, so they keep the mode).
    """

    def _open(self):
        """
        Opens the log file as the base class does, then restricts its permissions.

        :return: The open stream.
        """
        stream = super()._open()
        try:
            os.chmod(self.baseFilename, 0o660)
        except OSError:
            pass    # not ours to change (another user created it); the directory's permissions still apply
        return stream


def add_log_file(path: Optional[str], retention_days: int = LOG_RETENTION_DAYS, log_format: str = LOG_FORMAT,
                 logger: Optional[logging.Logger] = None) -> Optional[logging.Handler]:
    """
    Sends everything logged to the screen to a file as well (see the module docstring).

    Calling it again with the same file does nothing, so a server that loads its settings twice does not write every
    line twice. A file that cannot be opened (a missing permission, say) is reported on the screen, and the server
    carries on logging to the screen only - a log file is not worth refusing to start over.

    :param path: The log file; '~' is expanded and missing folders are created. None or '' means screen only.
    :param retention_days: How many days of files are kept (one per day).
    :param log_format: The line format (the console's, by default).
    :param logger: The logger to attach to; the root logger by default, which every module's logger reports to.
    :return: The handler that was added (or was already there), or None if nothing was added.
    """
    if not path:
        return None
    target = logger or logging.getLogger()
    path = os.path.abspath(os.path.expanduser(path))

    # Already writing to this file?
    for handler in target.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, 'baseFilename', None) == path:
            return handler

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = PrivateTimedRotatingFileHandler(path, when='midnight', backupCount=retention_days, encoding='utf-8')
    except OSError as e:
        target.error(f"Could not open log file {path} ({e}); logging to the screen only.")
        return None

    handler.setFormatter(PlainFormatter(log_format))
    # The file gets whatever the logger lets through - the same lines as the screen (basicConfig's level)
    target.addHandler(handler)
    target.info(f"Logging to {path} as well as the screen (one file a day, {retention_days} days kept).")
    return handler
