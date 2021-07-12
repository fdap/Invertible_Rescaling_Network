## 模型目的
Downscaling part of the IRN. You can train this sub-model by both **PYNATIVE** mode and **GRAPH** mode. 


## 参考实现
https://github.com/pkuxmq/Invertible-Image-Rescaling


## 代码目录结构
```
├── fig                       # 实例图像
│   ├── gt.jpg
│   ├── lq.jpg
│   └── out_lq.jpg            # 输出Low Quality 图像
├── src
│   ├── data                  # 数据集
│   │   ├── __init__.py
│   │   ├── LQGT_dataset.py
│   │   └── util.py
│   ├── network               # 网络结构
│   │   ├── __init__.py
│   │   ├── Invnet.py         # Invertible Rescaling Network
│   │   ├── net_with_loss.py  # 将loss和网络结合
│   │   └── util.py           
│   ├── optim                 # 学习率
│   │   ├── __init__.py
│   │   ├── warmup_cosine_annealing_lr.py
│   │   └── warmup_multisteplr.py
│   ├── options               # 模型配置选择
│   │   ├── __init__.py
│   │   ├── options.py
│   │   ├── test
│   │   │   ├── test_IRN_x2.yml
│   │   │   ├── test_IRN+_x4.yml
│   │   │   └── test_IRN_x4.yml
│   │   └── train
│   │       ├── train_IRN_x2.yml
│   │       ├── train_IRN+_x4.yml
│   │       └── train_IRN_x4.yml
│   └── utils               
│       ├── __init__.py
│       └── util.py
├── run_scripts.sh
└── train.py


```
