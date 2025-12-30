# TensorFlow 入门项目

本项目帮助你快速入门 TensorFlow 神经网络库。

## 环境配置

### 1. 激活虚拟环境

```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖（如需重新安装）

```bash
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行演示程序

```bash
python src/hello_tensorflow.py
```

## TensorFlow 工作流程

### 核心概念

1. **张量 (Tensor)** - TensorFlow 的基本数据结构，本质是多维数组
   - 标量：0维张量
   - 向量：1维张量
   - 矩阵：2维张量

2. **Eager Execution** - TensorFlow 2.x 默认的即时执行模式
   - 操作立即执行并返回结果
   - 便于调试和学习

3. **Graph Execution** - 使用 `@tf.function` 将函数编译为计算图
   - 性能更高
   - 适合生产环境

### 基本工作流程

```
定义数据/张量 -> 定义操作 -> 执行计算 -> 获取结果
```

## 项目结构

```
tensorflow-quickstart/
├── venv/                    # Python 虚拟环境
├── src/
│   └── hello_tensorflow.py  # TensorFlow 演示程序
├── requirements.txt         # 项目依赖
└── README.md               # 项目说明
```

## 下一步学习

- `tf.keras` - 构建神经网络模型
- `tf.data` - 数据处理管道
- `tf.GradientTape` - 自动微分
