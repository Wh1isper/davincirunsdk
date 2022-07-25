#  Copyright (c) 2022 Wh1isper
#
#  Use of this source code is governed by an MIT-style
#  license that can be found in the LICENSE file or at
#  https://opensource.org/licenses/MIT.
#
import errno
import os

from davincirunsdk.common import ModelArtsLog

_log = None


def init_log():
    global _log
    if _log:
        return _log
    log = ModelArtsLog.setup_modelarts_logger()
    _log = log
    return log


def is_in_notebook():
    """To check is in notenook or not. """

    try:
        import ipykernel
        ipykernel.get_connection_info()
    # Temporary fix for #84
    # TODO: remove blanket Exception catching after fixing #84
    except Exception:
        return False
    return True

def fsync_dir(dir_path):
    """
    Execute fsync on a directory ensuring it is synced to disk

    :param str dir_path: The directory to sync
    :raise OSError: If fail opening the directory
    """
    dir_fd = os.open(dir_path, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    except OSError as e:
        # On some filesystem doing a fsync on a directory
        # raises an EINVAL error. Ignoring it is usually safe.
        if e.errno != errno.EINVAL:
            raise
    finally:
        os.close(dir_fd)