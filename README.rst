davincirun
===============================

author: Wh1isper

Overview
--------

davincirun

华为多机多卡SDK，这里我们进行了二方库封装，并封装了一系列程序

davincirun.py是启动脚本，华为官方用法为

```bash
$davincirun python train.py
```

脚本做了如下工作：

- 初始化rank table(hccl json)

  - 根据k8s提供的`/user/config/jobstart_hccl.json`生成`/home/ma-user/rank_table/jobstart_hccl.json`

  - 设置环境变量`RANK_TABLE_FILE=/home/ma-user/rank_table/jobstart_hccl.json`

- 启动多进程训练
  - 根据上述生成的rank table，在/home/ma-user/下创建workspace目录，workspace目录下device{id}是各个device对应进程的实际工作目录

- 监控多进程状态

此SDK拆分上述工作为各个模块，其中初始化rank_table为全量入口调用，启动多进程训练并监控由数据分析师自主调用，实现分布式训练能力


Installation / Usage
--------------------

To install use pip:

    $ pip install davincirunsdk

Or clone the repo:

    $ git clone https://github.com/Wh1isper/davincirunsdk.git
    $ python setup.py install

Contributing
------------

TBD

Example
-------

TBD