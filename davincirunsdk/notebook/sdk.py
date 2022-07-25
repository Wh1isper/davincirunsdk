#  Copyright (c) 2022 Wh1isper
#
#  Use of this source code is governed by an MIT-style
#  license that can be found in the LICENSE file or at
#  https://opensource.org/licenses/MIT.
#


import os
import shutil
import uuid
from contextlib import contextmanager
from typing import Dict

from davincirunsdk.common import RankTableEnv, ModelArts
from davincirunsdk.notebook.manager import AscendVersionManager, Manager, FMKManager
from davincirunsdk.notebook.tailer import TailManager
from davincirunsdk.notebook.utils import init_log, fsync_dir
from davincirunsdk.rank_table import RankTable, RankTableV1, RankTableV0


def generate_rank_table():
    rank_table_path = os.environ.get(RankTableEnv.RANK_TABLE_FILE_V1)
    if rank_table_path:
        rank_table_cls = RankTableV1
    else:
        rank_table_path = RankTableEnv.get_rank_table_file_path()
        rank_table_cls = RankTableV0
    try:
        RankTable.wait_for_available(rank_table_path)
    except FileNotFoundError:
        rank_table = None
    else:
        rank_table = rank_table_cls(rank_table_path)
    return rank_table


def get_rank_table():
    try:
        rank_table_path = os.environ[RankTableEnv.RANK_TABLE_FILE]
    except KeyError:
        raise RuntimeError('No environment variable RANK_TABLE_FILE, try generate_rank_table() first.')
    return RankTableV1(rank_table_path)


def _set_extra_env(rank_table):
    # set extra env for V1 job
    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    rank_start = server['device'][0]['rank_id']
    os.environ[ModelArts.MA_CURRENT_HOST_IP] = instance.server_id
    os.environ['RANK_START'] = str(rank_start)
    os.environ['RANK_SIZE'] = str(rank_table.get_device_num())


def set_rank_env(rank_table):
    log = init_log()
    AscendVersionManager.print_ascend_driver_version()
    if not AscendVersionManager.is_atlas_c75_tr5():
        log.info('you are advised to use ASCEND_DEVICE_ID env instead of DEVICE_ID,'
                 ' as the DEVICE_ID env will be discarded in later versions')
        log.info('particularly, ${ASCEND_DEVICE_ID} == ${DEVICE_ID}, it\'s the logical device id')
    RankTableEnv.set_rank_table_env(rank_table.get_rank_table_path())
    _set_extra_env(rank_table)


def init_rank_table() -> Dict or False:
    log = init_log()
    log.info('Try to config Rank table file for ascend applications...')
    rank_table = generate_rank_table()
    if not rank_table:
        log.info('No rank table to init, Skip...')
        return False
    set_rank_env(rank_table)
    return os.environ.copy()


@contextmanager
def set_random_ms_cache_dir():
    log = init_log()
    log.info('Changing MindSpore Cache dir...')
    cache_dir = f'/cache/{uuid.uuid4()}'
    old_cache_dir = os.environ.get('MS_COMPILER_CACHE_PATH')
    try:
        try:
            os.environ['MS_COMPILER_CACHE_PATH'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            fsync_dir(cache_dir)
        except Exception as e:
            log.warning('Fail to setup cache dir, will using default')
            log.exception(e)
            yield
            return

        yield
    finally:
        if old_cache_dir:
            os.environ['MS_COMPILER_CACHE_PATH'] = old_cache_dir
        else:
            del os.environ['MS_COMPILER_CACHE_PATH']
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            log.warning('Fail to cleanup cache dir')
            log.exception(e)


def wait_distributed_train(fmk_manager, destroy_when_finished=True, raise_exception=True):
    return fmk_manager.wait(destroy_when_finished, raise_exception)


def start_distributed_train(command, work_dir='./', log_dir='./log', *, output_notebook=False):
    init_log()
    rank_table = get_rank_table()
    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    current_instance = RankTable.convert_server_to_instance(server)
    fmk_manager = FMKManager(current_instance)
    fmk_manager.run(rank_table.get_device_num(), command, work_dir, log_dir, output_notebook=output_notebook)
    return fmk_manager


def start_and_wait_distributed_train(command, work_dir='./', log_dir='./log', *, output_notebook=False,
                                     random_cache_dir=True, destroy_when_finished=True, raise_exception=True):
    def _run_wait():
        fmk_manager.run(rank_table.get_device_num(), command, work_dir, log_dir, output_notebook=output_notebook)
        return fmk_manager.wait(destroy_when_finished, raise_exception)

    init_log()
    rank_table = get_rank_table()
    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    current_instance = RankTable.convert_server_to_instance(server)
    fmk_manager = FMKManager(current_instance)
    if random_cache_dir:
        with set_random_ms_cache_dir():
            return _run_wait()
    else:
        return _run_wait()


if __name__ == '__main__':
    init_rank_table()
