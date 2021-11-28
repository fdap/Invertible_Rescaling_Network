## 目录

- [目录](#目录)
- [网络描述](#网络描述)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
- [脚本参数](#脚本参数)
- [脚本使用](#脚本使用)
    - [训练脚本用法](#训练脚本用法)
    - [评估脚本用法](#评估脚本用法)
    - [导出脚本用法](#导出脚本用法)
- [模型描述](#模型描述)
- [随机情况说明](#随机情况说明)
- [官方主页](#官方主页)

## 网络描述

### 概述

高分辨率数字图像通常被缩小以适应各种显示屏幕或节省存储和带宽成本，同时采用后放大来恢复原始分辨率或放大图像中的细节。

然而，典型的图像降尺度由于高频信息的丢失是一种非注入映射，这导致逆升尺度过程的不适定问题，并对从降尺度的低分辨率图像中恢复细节提出了巨大的挑战。简单地使用图像超分辨率方法进行放大会导致无法令人满意的恢复性能。

可逆重缩放网络 (IRN)从新的角度对缩小和放大过程进行建模来解决这个问题，即可逆双射变换，在缩小过程中使用遵循特定分布的潜在变量来捕获丢失信息的分布，这可以在很大程度上减轻图像放大的不适定性质,以生成视觉上令人愉悦的低分辨率图像。通过这种方式，通过网络将随机绘制的潜在变量与低分辨率图像反向传递，从而使放大变得易于处理。

本示例主要针对IRN提出的深度神经网络架构以及训练过程进行了实现。

### 论文

Mingqing Xiao, Shuxin Zheng, Chang Liu, Yaolong Wang, Di He, Guolin Ke, Jiang Bian, Zhouchen Lin, and Tie-Yan Liu. 2020. Invertible Image Rescaling. In European Conference on Computer Vision (ECCV).

## 模型架构

![1](./figures/architecture.jpg)

## 数据集

本示例使用[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)，其目录结构如下：

```bash
DIV2K_data
├── DIV2K_train_HR/                 # 训练集数据
└── DIV2K_valid_HR/                 # 测试集数据
```

## 环境要求

- 硬件
    - GPU


## 快速入门

完成计算设备和框架环境的准备后，开发者可以运行如下指令对本示例进行训练和评估。

- GPU环境运行

```bash
# 单卡训练
用法：bash run_standalone_train.sh [SCALE] [DATASET_GT_PATH]

# 单卡评估
用法：bash run_eval.sh [SCALE] [DATASET_PATH] [CHECKPOINT_PATH]
```

## 脚本说明

```bash
.
├── README.md                               # 说明文档
├── src
│   ├── data
│   │   ├── LQGT_dataset.py                 # 数据集处理
│   │   └── util.py                         # 数据集读取图片缩放等
│   ├── network
│   │   ├── Invnet.py                       # IRN网络定义
│   │   ├── net_with_loss.py                # 自定义loss
│   │   └── util.py                         # 网络初始化等
│   ├── optim
│   │   ├── adam_clip.py                    # 梯度裁剪
│   │   ├── warmup_cosine_annealing_lr.py   # 余弦退火学习率算法
│   │   └── warmup_multisteplr.py           # 多步学习率算法
│   ├── options
│   │   ├── options.py                      # 配置文件读取
│   │   ├── test
│   │   │   ├── test_IRN_x2.yml             # 2倍缩放测试配置文件
│   │   │   └── test_IRN_x4.yml             # 4倍缩放测试配置文件
│   │   └── train
│   │       ├── train_IRN_x2.yml            # 2倍缩放训练配置文件
│   │       └── train_IRN_x4.yml            # 4倍缩放训练配置文件
│   └── utils
│       └── util.py                         # 评价指标计算
├── train.py                                # 训练网络
└── val.py                                  # 测试网络
```

## 脚本参数

在[/src/options/train/train_IRN_x4.yml](./src/options/train/train_IRN_x4.yml)中可以配置训练参数、数据集路径等参数。


## 脚本使用

### 训练脚本用法

```bash
python train.py -opt ./src/options/train/train_IRN_x4.yml
```

### 评估脚本用法

对训练好的模型进行精度评估：

```bash
python val.py -opt ./src/options/test/test_IRN_x4.yml
```


## 模型描述

| 参数 | 单卡GPU | 单卡Ascend 910 | 8卡Ascend 910 |
|:---|:---|:---|:--|
| 资源 | GTX 1080ti | Ascend 910 | Ascend 910|
| 上传日期 | 2021.09.25 | 2021.09.25 | 2021.11.01 |
| MindSpore版本 | 1.2.0 | 1.3.0 | 1.3.0 |
| 训练数据集 | DIV2K | DIV2K | DIV2K |
| 优化器 | Adam | Adam | Adam |
| 输出 | Reconstructed HR image | Reconstructed HR image | Reconstructed HR image |
| PSNR | 34.83 | 34.11 | 33.88 |
| SSIM | 0.9287  | 0.9206 | 0.9167 |
| 速度 | 1534 ms/step | 271 ms/step | 409 ms/step |
| 总时长 | 3162 mins | 2258 mins | 409 mins
| 微调检查点 | 50.1M（.ckpt文件) | 50.1M（.ckpt文件) | 50.1M（.ckpt文件) |
| 脚本 | [IRN](./) | [IRN](./) | [IRN](./) |

## 随机情况说明

[train.py](./train.py)中设置了随机种子，以确保训练的可复现性。

## 官方主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
