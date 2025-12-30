# 🏥 糖尿病视网膜病变图像分类 - 集成学习系统

## 📋 项目简介

本项目使用 **8个深度学习模型** 进行集成学习，对糖尿病视网膜病变（Diabetic Retinopathy）图像进行 **5分类**：

| 类别 | 英文名 | 中文名 |
|------|--------|--------|
| 0 | No_DR | 无病变 |
| 1 | Mild | 轻度 |
| 2 | Moderate | 中度 |
| 3 | Severe | 重度 |
| 4 | Proliferative | 增殖性 |

---

## 👥 团队分工

| 成员 | 负责模型 | Git分支 |
|------|----------|---------|
| **组长** | ResNet-50, EfficientNet-B0 | `model-resnet-efficientnet` |
| **成员A** | VGG-16, MobileNetV2 | `model-vgg-mobilenet` |
| **成员B** | SE-ResNet, ResNeXt-50 | `model-seresnet-resnext` |
| **成员C** | DenseNet-121, InceptionV3 | `model-densenet-inception` |

---

## 📁 项目结构详解

```
diabetic-retinopathy/
│
├── 📄 README.md                    # 👈 你正在看的文件
├── 📄 requirements.txt             # Python依赖包列表
├── 📄 .gitignore                   # Git忽略配置
│
├── 📂 docs/                        # 📚 文档目录
│   ├── 项目整体规划.md             # 完整的项目规划和操作指南
│   └── 团队任务说明.md             # Git操作和任务清单
│
├── 📂 src/                         # 🔧 源代码目录
│   ├── config.py                   # ⚙️ 全局配置文件
│   │
│   ├── 📂 data/                    # 数据加载模块
│   │   ├── __init__.py
│   │   └── data_loader.py          # 🔒 数据加载器（不要修改）
│   │
│   ├── 📂 models/                  # 🎯 模型定义（你需要在这里添加文件）
│   │   ├── __init__.py
│   │   ├── base_model.py           # 🔒 模型基类（不要修改）
│   │   ├── resnet50.py             # 📝 示例模型（参考这个写）
│   │   ├── vgg16.py                # ✏️ 成员A创建
│   │   ├── mobilenetv2.py          # ✏️ 成员A创建
│   │   ├── se_resnet.py            # ✏️ 成员B创建
│   │   ├── resnext50.py            # ✏️ 成员B创建
│   │   ├── densenet121.py          # ✏️ 成员C创建
│   │   └── inceptionv3.py          # ✏️ 成员C创建
│   │
│   ├── 📂 training/                # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py              # 🔒 训练器（不要修改）
│   │   └── callbacks.py            # 🔒 回调函数（不要修改）
│   │
│   ├── 📂 evaluation/              # 评估模块
│   │   ├── __init__.py
│   │   └── evaluator.py            # 🔒 评估器（不要修改）
│   │
│   ├── 📂 ensemble/                # 集成学习模块
│   │   ├── __init__.py
│   │   ├── ensemble_model.py       # 🔒 集成模型（不要修改）
│   │   └── voting.py               # 🔒 投票策略（不要修改）
│   │
│   └── 📂 preprocessing/           # 数据预处理（已完成）
│       ├── __init__.py
│       ├── dataset_splitter.py     # 🔒 数据集划分
│       ├── class_balancer.py       # 🔒 类别平衡
│       └── augmentation/           # 🔒 数据增强
│
├── 📂 scripts/                     # 🚀 运行脚本
│   ├── train_model.py              # 训练脚本
│   ├── evaluate_model.py           # 评估脚本
│   └── run_ensemble.py             # 集成学习脚本
│
├── 📂 trained_models/              # 💾 训练好的模型保存位置
│   ├── resnet50/                   # 组长的模型
│   ├── efficientnet_b0/            # 组长的模型
│   ├── vgg16/                      # 成员A的模型
│   ├── mobilenetv2/                # 成员A的模型
│   ├── se_resnet/                  # 成员B的模型
│   ├── resnext50/                  # 成员B的模型
│   ├── densenet121/                # 成员C的模型
│   └── inceptionv3/                # 成员C的模型
│
└── 📂 output/                      # 📊 输出目录
    └── preprocessing/              # 预处理结果（已生成）
        ├── train.csv               # 训练集文件列表
        ├── val.csv                 # 验证集文件列表
        └── test.csv                # 测试集文件列表
```

---

## 🔒 文件权限说明

### ❌ 不要修改的文件（公共模块）

这些文件是所有人共用的基础设施，修改会导致冲突：

| 文件 | 作用 | 原因 |
|------|------|------|
| `src/config.py` | 全局配置 | 统一的参数设置 |
| `src/data/data_loader.py` | 数据加载 | 确保所有人用相同的数据处理 |
| `src/models/base_model.py` | 模型基类 | 所有模型的父类 |
| `src/training/trainer.py` | 训练器 | 统一的训练流程 |
| `src/training/callbacks.py` | 回调函数 | 统一的训练回调 |
| `src/evaluation/evaluator.py` | 评估器 | 统一的评估标准 |
| `src/ensemble/*` | 集成模块 | 组长负责 |
| `src/preprocessing/*` | 预处理 | 已完成，不需要改 |
| `scripts/*` | 运行脚本 | 统一的运行方式 |

### ✅ 需要创建/修改的文件

| 成员 | 需要创建的文件 |
|------|---------------|
| 成员A | `src/models/vgg16.py`, `src/models/mobilenetv2.py` |
| 成员B | `src/models/se_resnet.py`, `src/models/resnext50.py` |
| 成员C | `src/models/densenet121.py`, `src/models/inceptionv3.py` |

### 📁 需要上传的输出文件

训练完成后，每个模型目录下应该有：

```
trained_models/你的模型名/
├── 你的模型名_best.keras      # ✅ 必须上传 - 模型权重
├── training_history.json      # ✅ 必须上传 - 训练历史
├── evaluation_report.json     # ✅ 必须上传 - 评估报告
├── confusion_matrix.png       # ✅ 必须上传 - 混淆矩阵图
└── metadata.json              # ✅ 必须上传 - 元数据
```


---

## 🚀 快速开始

### 第一步：克隆项目

```bash
git clone <仓库地址>
cd diabetic-retinopathy
```

### 第二步：安装依赖

```bash
pip install -r requirements.txt
```

### 第三步：准备数据

将数据集放到以下位置（数据不上传Git，每人本地准备）：

```
D:\machine_learning\data\data2\
├── 0/    # 无病变图片
├── 1/    # 轻度图片
├── 2/    # 中度图片
├── 3/    # 重度图片
└── 4/    # 增殖性图片
```

### 第四步：创建你的分支

```bash
# 确保在main分支
git checkout main
git pull origin main

# 创建你的分支（根据你负责的模型选择）
git checkout -b model-vgg-mobilenet        # 成员A
git checkout -b model-seresnet-resnext     # 成员B
git checkout -b model-densenet-inception   # 成员C
```

### 第五步：创建你的模型文件

在 `src/models/` 目录下创建你的模型文件，**参考 `resnet50.py`**：

```python
# src/models/vgg16.py （示例）
from .base_model import BaseModel
from tensorflow import keras
from tensorflow.keras import layers

class VGG16Model(BaseModel):
    """VGG-16 模型"""
    
    def __init__(self, num_classes: int = 5,
                 input_shape: tuple = (224, 224, 3),
                 pretrained: bool = True):
        super().__init__(
            model_name='vgg16',  # ⚠️ 改成你的模型名
            num_classes=num_classes,
            input_shape=input_shape
        )
        self.pretrained = pretrained
    
    def build(self) -> keras.Model:
        # 1. 加载预训练模型
        base_model = keras.applications.VGG16(  # ⚠️ 改成你的模型
            weights='imagenet' if self.pretrained else None,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # 2. 冻结预训练层
        base_model.trainable = False
        
        # 3. 构建完整模型
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
```

### 第六步：训练模型

```bash
# 训练（替换为你的模型名）
python scripts/train_model.py --model vgg16 --epochs 50 --batch_size 32

# 如果GPU内存不足，减小batch_size
python scripts/train_model.py --model vgg16 --epochs 50 --batch_size 16
```

### 第七步：评估模型

```bash
python scripts/evaluate_model.py --model vgg16
```

### 第八步：提交代码

```bash
# 添加更改
git add .

# 提交（写清楚做了什么）
git commit -m "feat: 完成VGG-16模型训练，准确率82%"

# 推送
git push origin model-vgg-mobilenet  # 替换为你的分支名
```

### 第九步：创建Pull Request

1. 打开GitHub仓库页面
2. 点击 "Pull requests" → "New pull request"
3. 选择你的分支合并到main
4. 等待组长审核

---

## 📝 各模型实现参考

### 成员A：VGG-16 和 MobileNetV2

```python
# VGG-16
base_model = keras.applications.VGG16(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)

# MobileNetV2
base_model = keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
```

### 成员B：SE-ResNet 和 ResNeXt-50

**SE模块实现：**
```python
def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation模块"""
    channels = input_tensor.shape[-1]
    
    # Squeeze
    x = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation
    x = layers.Dense(channels // ratio, activation='relu')(x)
    x = layers.Dense(channels, activation='sigmoid')(x)
    
    # Scale
    x = layers.Reshape((1, 1, channels))(x)
    return layers.Multiply()([input_tensor, x])
```

### 成员C：DenseNet-121 和 InceptionV3

```python
# DenseNet-121
base_model = keras.applications.DenseNet121(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)

# ⚠️ InceptionV3 - 注意输入尺寸是 299x299！
base_model = keras.applications.InceptionV3(
    weights='imagenet', include_top=False, input_shape=(299, 299, 3)
)
```

**InceptionV3特殊处理：**
```python
class InceptionV3Model(BaseModel):
    def __init__(self):
        super().__init__(
            model_name='inceptionv3',
            input_shape=(299, 299, 3)  # ⚠️ 特殊尺寸！
        )
```

---

## ❓ 常见问题

### Q1: GPU内存不足？
```bash
# 减小batch_size
python scripts/train_model.py --model vgg16 --batch_size 8
```

### Q2: 训练太慢？
- 使用GPU训练
- 减少epochs，依赖early stopping
- 先用小数据集测试流程

### Q3: 模型准确率太低？
- 增加训练轮数
- 调整学习率（试试0.0001）
- 解冻部分预训练层进行微调

### Q4: Git冲突？
```bash
git pull origin main
# 手动解决冲突后
git add .
git commit -m "fix: 解决合并冲突"
```

---

## ✅ 提交检查清单

提交PR前，确认以下文件都存在：

- [ ] `src/models/你的模型.py` - 模型定义文件
- [ ] `trained_models/你的模型/你的模型_best.keras` - 模型权重
- [ ] `trained_models/你的模型/training_history.json` - 训练历史
- [ ] `trained_models/你的模型/evaluation_report.json` - 评估报告
- [ ] `trained_models/你的模型/confusion_matrix.png` - 混淆矩阵
- [ ] 模型准确率 > 70%

---

## 📊 预期结果

| 模型 | 预期准确率 |
|------|-----------|
| ResNet-50 | 80-85% |
| EfficientNet-B0 | 82-87% |
| VGG-16 | 78-83% |
| MobileNetV2 | 75-80% |
| SE-ResNet | 82-86% |
| ResNeXt-50 | 81-85% |
| DenseNet-121 | 80-84% |
| InceptionV3 | 79-83% |
| **集成模型** | **85-90%** |

---

## 📞 遇到问题？

1. 先查看 `docs/项目整体规划.md` 和 `docs/团队任务说明.md`
2. 参考示例代码 `src/models/resnet50.py`
3. 在群里讨论

祝大家顺利完成项目！🎉
