![](https://img.shields.io/github/license/wh1isper/davincirunsdk) 
![](https://img.shields.io/github/v/release/wh1isper/davincirunsdk) 
![](https://img.shields.io/pypi/dm/davincirunsdk) 
![](https://img.shields.io/github/last-commit/wh1isper/davincirunsdk)
![](https://img.shields.io/pypi/pyversions/davincirunsdk)
![](https://readthedocs.org/projects/davincirunsdk/badge/?version=latest)

# davincirunsdk

为类Jupyter交互式环境提供Notebook友好的Ascend分布式训练SDK，推荐在[AI靶场](https://git.openi.org.cn/datai/datai_tutorial)使用本脚本进行python脚本训练

## 特性

- `davincirun`命令，支持Modelarts Ascend训练作业，不再需要打包davinci文件夹
- `init_rank_table` 支持转换v0.1 hccl json -> v1.0 hccl json
- `start_distributed_train`, `wait_distributed_train`  根据v1.0 hccl json启动并等待分布式训练完成
- notebook友好，`output_notebook=True`支持在notebook中输出分布式训练日志

更多见[SDK文档](http://davincirunsdk.readthedocs.io/)

![](./image/example.gif)

## 安装及使用

### 安装

`$pip install davincirunsdk`

### 调试环境（开发环境）

以[MindSpore1.5分布式训练教程](https://www.mindspore.cn/tutorials/zh-CN/r1.5/intermediate/distributed_training/distributed_training_ascend.html) 为例，使用本SDK可改造为

```python
import os

os.environ['DATA_PATH'] = '/cache/cifar-10-batches-bin'
from davincirunsdk import start_and_wait_distributed_train

cmd = ['python', 'resnet50_distributed_training.py']
start_and_wait_distributed_train(cmd, output_notebook=True)
```

### 训练作业

以下命令将等价于`python davincirun.py train.py`

```bash
$davincirun train.py
```

或在python文件中使用：

```python
from davincirunsdk import init_rank_table, start_and_wait_distributed_train

init_rank_table()
start_and_wait_distributed_train(['python', 'train.py'])
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
$pip install pytest
$pytest .
```

### 文档构建

```bash
$pip install Sphinx sphinx-rtd-theme
$make html
```


### 项目构成

`notebook`文件夹下是针对notebook运行环境修改的davincirun文件，以及sdk入口

`davincirunsdk`目录下，除了`notebook`外的文件，是原有davincurun代码，进行了python包改造，并按需启用了moxing对obs文件的支持

各项文档和API功能还在完善中，欢迎各位在issue中进行反馈

## 案例

[AI靶场分布式训练支持](https://git.openi.org.cn/Wh1isper/distrubuted-trainning-on-datai)

[盘古协同计算](https://git.openi.org.cn/PCL-Platform.Intelligence/AISynergy/src/branch/V2.0.0/examples/Pangu-AISync)

## 鸣谢

感谢华为云、鹏城实验室、AI靶场对本项目的大力支持和帮助，该项目已贡献给[AI靶场](https://git.openi.org.cn/datai)

# LINK

不如在这里给我们一个Star🌟鼓励一下！ 

🌟🌟🌟[Github](https://github.com/Wh1isper/davincirunsdk) 🌟🌟🌟

🌟🌟🌟[OpenI](https://git.openi.org.cn/Wh1isper/davincirunsdk) 🌟🌟🌟

