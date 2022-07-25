# davincirunsdk

为类Jupyter交互式环境提供Notebook友好的Ascend分布式训练SDK

## 安装及使用

### 安装

`$pip install davincirunsdk`

### 调试环境（开发环境）

以[MindSpore1.5分布式训练教程](https://www.mindspore.cn/tutorials/zh-CN/r1.5/intermediate/distributed_training/distributed_training_ascend.html) 为例，使用本SDK可改造为

```python
import os

os.environ['DATA_PATH'] = '/cache/cifar-10-batches-bin'
from davincirunsdk import start_distributed_train, wait_distributed_train

cmd = ['python', 'resnet50_distributed_training.py']
manager = start_distributed_train(cmd, output_notebook=True)
wait_distributed_train(manager)
```

### 训练作业

以下命令将等价于`python davincirun.py train.py`

```bash
$davincirun train.py
```

或在python文件中使用：

```python
from davincirunsdk import init_rank_table, start_distributed_train, wait_distributed_train

init_rank_table()
manager = start_distributed_train(['python', 'train.py'])
wait_distributed_train(manager)
```

### AI靶场全量运行

同[调试环境（开发环境）](#调试环境（开发环境）)，不需要额外修改

更多细节见[案例](#案例)

## LICENSE

MIT License

## 开发指南

### 克隆并安装

```bash
$git clone https://git.openi.org.cn/Wh1isper/davincirunsdk.git
$cd davincirunsdk
$pip install -e ./
```

### 单元测试

```bash
$pytest .
```

### 项目构成

`notebook`文件夹下是针对notebook运行环境修改的davincirun文件，以及sdk入口

`davincirunsdk`目录下，除了`notebook`外的文件，是原有davincurun代码，进行了python包改造，并按需启用了moxing对obs文件的支持

## 案例

[AI靶场分布式训练支持](https://git.openi.org.cn/Wh1isper/distrubuted-trainning-on-datai)

## 鸣谢

感谢华为云、鹏城实验室、AI靶场对本项目的大力支持和帮助