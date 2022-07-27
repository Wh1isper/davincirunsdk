.. davincirunsdk documentation master file, created by
   sphinx-quickstart on Tue Jul 26 15:55:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to davincirunsdk's documentation!
=========================================

为类Jupyter交互式环境提供Notebook友好的Ascend分布式训练SDK

* ``davincirun`` 命令，支持Modelarts Ascend训练作业，不再需要打包davinci文件夹
* ``init_rank_table`` 支持转换v0.1 hccl json -> v1.0 hccl json
* ``start_distributed_train`` , ``wait_distributed_train``  根据v1.0 hccl json启动并等待分布式训练完成
* notebook友好，``output_notebook=True`` 支持在notebook中输出分布式训练日志


Installation
============

.. code-block:: bash

    pip install davincirunsdk

仓库地址
============

* `Github Repo <https://github.com/wh1isper/davincirunsdk>`_
* `OpenI <https://git.openi.org.cn/Wh1isper/davincirunsdk>`_

Reference
========================================

.. toctree::
   :maxdepth: 3
   :caption: 参考API文档

   公共API <public/index>
   notebook <notebook/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
