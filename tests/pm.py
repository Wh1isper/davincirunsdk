import os
import re

import papermill
from papermill import PapermillExecutionError


def remove_ansi_escape(text):
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    ''', re.VERBOSE)
    return ansi_escape.sub('', text)


dir_prefix = os.path.dirname(__file__)
input_file_name = os.path.join(dir_prefix, 'pm.ipynb')
output_file = os.path.join(dir_prefix, 'pm-out.ipynb')
try:
    papermill.execute_notebook(
        input_file_name,
        output_file,
        log_output=True,
    )
except PapermillExecutionError as e:
    if e.ename == 'DistributedRuntimeError':
        traceback = '\n'.join(e.traceback[:])
    else:
        traceback = '\n'.join(e.traceback[:-1])
    if traceback:
        raise RuntimeError(remove_ansi_escape(traceback))
