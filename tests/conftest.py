import pytest
import os
import shutil

from davincirunsdk.common import RankTableEnv

cache_dir = '/cache'
dir_prefix = os.path.dirname(__file__)
generated_hccl_path = '/home/ma-user/rank_table/jobstart_hccl.json'
k8s_hccl_path = RankTableEnv.get_rank_table_file_path()
mock_k8s_hccl_file = os.path.join(dir_prefix, 'k8s_jobstart_hccl.json')


def setup():
    try:
        os.makedirs(os.path.dirname(generated_hccl_path), exist_ok=True)
    except OSError:
        raise OSError(f'Cant mkdir for cache: {cache_dir}')
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


@pytest.fixture(autouse=True)
def setup_and_cleanup():
    setup()
    yield
    cleanup()
