import os
from typing import Dict

from davincirunsdk.common import RankTableEnv, ModelArts
from davincirunsdk.customize.manager import AscendVersionManager, Manager, FMKManager
from davincirunsdk.customize.tailer import TailManager
from davincirunsdk.customize.utils import init_log
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


def start_distributed_train(command, work_dir='./', log_dir='./log', *, output_notebook=False):
    rank_table = get_rank_table()
    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    current_instance = RankTable.convert_server_to_instance(server)
    fmk_manager = FMKManager(current_instance)
    fmk_manager.run(rank_table.get_device_num(), command, work_dir, log_dir, output_notebook=output_notebook)
    return fmk_manager


def wait_distributed_train(fmk_manager, destroy_when_finished=True, raise_exception=True):
    return fmk_manager.wait(destroy_when_finished, raise_exception)


if __name__ == '__main__':
    init_rank_table()
