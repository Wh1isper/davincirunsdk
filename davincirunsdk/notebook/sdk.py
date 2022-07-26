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
from davincirunsdk.notebook.manager import AscendVersionManager, FMKManager
from davincirunsdk.notebook.utils import init_log, fsync_dir
from davincirunsdk.rank_table import RankTable, RankTableV1, RankTableV0


def generate_rank_table():
    """ 训练作业时用于hccl v0.1 -> v1.0转换，如果当前已经有了v1.0的hccl文件，直接使用get_rank_table

    Returns:
        RankTable, 可能是RankTableV0或RankTableV1
    """
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
    """ 读入环境变量中的RANK_TABLE

    Returns:
        RankTableV1
    """
    try:
        rank_table_path = os.environ[RankTableEnv.RANK_TABLE_FILE]
    except KeyError:
        raise RuntimeError('No environment variable RANK_TABLE_FILE, try generate_rank_table() first.')
    return RankTableV1(rank_table_path)


def _set_extra_env(rank_table):
    """ 训练任务转换hccl V0.1 -> v1.0时，额外适配的环境变量

    Args:
        rank_table: RankTable，可以是V0或者V1

    Returns:

    """
    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    rank_start = server['device'][0]['rank_id']

    # RankTableV1通过此环境变量获取当前机器IP，旧版训练任务未设置，此处补上
    os.environ[ModelArts.MA_CURRENT_HOST_IP] = instance.server_id

    # 为用户自主编写多进程脚本提供的环境变量
    os.environ['RANK_START'] = str(rank_start)
    os.environ['RANK_SIZE'] = str(rank_table.get_device_num())


def set_rank_env(rank_table):
    """ 这里重新设置了hccl文件的地址，主要是针对V0.1转换为V1.0转换的场景

    Args:
        rank_table: RankTable，可以是V0或者V1


    Returns:
    """

    log = init_log()
    AscendVersionManager.print_ascend_driver_version()

    # 原生warming，这里保留
    if not AscendVersionManager.is_atlas_c75_tr5():
        log.info('you are advised to use ASCEND_DEVICE_ID env instead of DEVICE_ID,'
                 ' as the DEVICE_ID env will be discarded in later versions')
        log.info('particularly, ${ASCEND_DEVICE_ID} == ${DEVICE_ID}, it\'s the logical device id')
    RankTableEnv.set_rank_table_env(rank_table.get_rank_table_path())
    _set_extra_env(rank_table)


def init_rank_table() -> Dict or False:
    """ SDK，训练作业中用户应使用此函数转换hccl v0.1 -> v1.0

    Returns:
        设置后的环境变量， False则为未找到rank_table，跳过设置
    """

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
    """ 用于暂时设置MindSpore compiler缓存文件夹，用完自动销毁；
        这个方法允许你在启动分布式训练后做一些额外的工作，如果不需要，可以使用start_and_wait_distributed_train

    Example:
        with set_random_ms_cache_dir():
            manager = start_distributed_train(train_command)
            ... # do some extra work
            wait_distributed_train(manager)

    Returns:
    """
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
    """

    Args:
        fmk_manager:  FMKManager, 通常是使用start_distributed_train的返回
        destroy_when_finished: 默认为True，是否在结束时销毁所有子进程；通常及时销毁可以帮助释放NPU资源，除非你想深入进程细节
        raise_exception: 默认为True，是否在子进程失败时raise exception，以确保外部得到exception提示，这在流水线中判断执行结果很有用

    Returns:
        状态码，0为正常结束，1为异常

    Raises:
        DistributedRuntimeError

    """

    return fmk_manager.wait(destroy_when_finished, raise_exception)


def start_distributed_train(command, work_dir='./', log_dir='./log', *, output_notebook=False):

    """
    Args:
        command (List) : command list，用于启动训练脚本
        work_dir: 工作目录，如果command存在相对路径，需要确保从工作目录访问相对路径正确
        log_dir: 日志输出目录
        output_notebook: 默认为False，当为True时，将自动输出日志到notebook中；如果在非notebook环境中打开，不应当有任何作用

    Example:
        with set_random_ms_cache_dir():
            manager = start_distributed_train(train_command)
            ... # do some extra work
            wait_distributed_train(manager)

    Returns:
        FMKManager

    """

    init_log()
    rank_table = get_rank_table()
    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    current_instance = RankTable.convert_server_to_instance(server)
    fmk_manager = FMKManager(current_instance)
    fmk_manager.run(rank_table.get_device_num(), command, work_dir, log_dir, output_notebook=output_notebook)
    return fmk_manager


def start_and_wait_distributed_train(command, work_dir='./', log_dir='./log',
                                     *,
                                     output_notebook=False,
                                     random_cache_dir=True,
                                     destroy_when_finished=True,
                                     raise_exception=True):
    """

    Args:
        command (List) : command list，用于启动训练脚本
        work_dir (Path-like string): 工作目录，如果command存在相对路径，需要确保从工作目录访问相对路径正确
        log_dir (Path-like string): 日志输出目录
        output_notebook: 默认为False，当为True时，将自动输出日志到notebook中；如果在非notebook环境中打开，不应当有任何作用
        random_cache_dir: 默认为True，是否使用随机缓存目录，避免在工作目录下生成大量算子缓存
        destroy_when_finished: 默认为True，是否在结束时销毁所有子进程；通常及时销毁可以帮助释放NPU资源，除非你想深入进程细节
        raise_exception: 默认为True，是否在子进程失败时raise exception，以确保外部得到exception提示，这在流水线中判断执行结果很有用

    Returns:
        状态码，0为正常结束，1为异常
    """

    def _run_wait():
        fmk_manager = start_distributed_train(
            command,
            work_dir=work_dir,
            log_dir=log_dir,
            output_notebook=output_notebook
        )
        return wait_distributed_train(
            fmk_manager,
            destroy_when_finished=destroy_when_finished,
            raise_exception=raise_exception
        )

    if random_cache_dir:
        with set_random_ms_cache_dir():
            return _run_wait()
    else:
        return _run_wait()


if __name__ == '__main__':
    init_rank_table()
