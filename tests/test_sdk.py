import pytest
import json
import os
import shutil

from davincirunsdk import init_rank_table, start_distributed_train, wait_distributed_train
from davincirunsdk.common import RankTableEnv
from davincirunsdk.notebook.exception import DistributedRuntimeError

dir_prefix = os.path.dirname(__file__)
generated_hccl_path = '/home/ma-user/rank_table/jobstart_hccl.json'
k8s_hccl_path = RankTableEnv.get_rank_table_file_path()
mock_k8s_hccl_file = os.path.join(dir_prefix, 'k8s_jobstart_hccl.json')
mock_train_file = os.path.join(dir_prefix, 'mock_train.py')
mock_failure_file = os.path.join(dir_prefix, 'mock_failure.py')


def test_init_rank_table():
    init_rank_table()
    assert os.path.exists(generated_hccl_path)
    with open(generated_hccl_path) as f:
        generated = json.load(f)
    with open(os.path.join(dir_prefix, 'generated_jobstart_hccl.json')) as f:
        right_generated = json.load(f)
    assert generated == right_generated


def test_full_stack():
    init_rank_table()
    manager = start_distributed_train(['python', mock_train_file])
    assert wait_distributed_train(manager) == 0


def test_failure_stack():
    init_rank_table()

    manager = start_distributed_train(['python', mock_failure_file])
    assert wait_distributed_train(manager, raise_exception=False) != 0

    manager = start_distributed_train(['python', mock_failure_file])
    with pytest.raises(DistributedRuntimeError):
        wait_distributed_train(manager)


if __name__ == '__main__':
    pytest.main()
