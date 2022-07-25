#  Copyright (c) 2022 Wh1isper
#
#  Use of this source code is governed by an MIT-style
#  license that can be found in the LICENSE file or at
#  https://opensource.org/licenses/MIT.
#


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
