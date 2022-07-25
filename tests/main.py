import pytest
import json
import os
import shutil

import wfio
from davincirunsdk import init_rank_table, start_distributed_train, wait_distributed_train
from davincirunsdk.common import RankTableEnv

dir_prefix = os.path.dirname(__file__)
generated_hccl_path = '/home/ma-user/rank_table/jobstart_hccl.json'
k8s_hccl_path = RankTableEnv.get_rank_table_file_path()
mock_k8s_hccl_file = os.path.join(dir_prefix, 'k8s_jobstart_hccl.json')
mock_train_file = os.path.join(dir_prefix, 'mock_train.py')
mock_failure_file = os.path.join(dir_prefix, 'mock_failure.py')


def setup():
    try:
        os.makedirs(os.path.dirname(generated_hccl_path), exist_ok=True)
    except OSError:
        raise OSError(f'Cant mkdir for generated_hccl_path: {generated_hccl_path}')
    try:
        os.makedirs(os.path.dirname(k8s_hccl_path), exist_ok=True)
    except OSError:
        raise OSError(f'Cant mkdir for k8s_hccl_path: {k8s_hccl_path}')

    try:
        os.remove(RankTableEnv.get_rank_table_file_path())
        os.remove(generated_hccl_path)
        os.remove(k8s_hccl_path)
    except FileNotFoundError:
        pass

    shutil.copy(mock_k8s_hccl_file, k8s_hccl_path)


def cleanup():
    del os.environ['RANK_TABLE_FILE']

    try:
        os.remove(k8s_hccl_path)
        os.remove(generated_hccl_path)
    except FileNotFoundError:
        pass


setup()

init_rank_table()
manager = start_distributed_train(['python', mock_failure_file])
try:
    wait_distributed_train(manager)
except Exception as e:
    all_log = str(e)
    with open('errlog.txt', 'w') as f:
        f.write(all_log)
    wfio.upload_to_oss('errlog.txt', 'errlog.txt')
    raise

cleanup()
