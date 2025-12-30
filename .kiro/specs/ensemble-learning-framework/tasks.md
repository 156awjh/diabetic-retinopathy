# Implementation Plan: 糖尿病视网膜病变集成学习系统

## Overview

本任务列表按照项目阶段划分，组长负责框架搭建和集成，团队成员并行训练各自负责的模型。

## Phase 1: 框架搭建（组长）

- [x] 1. 创建项目基础结构
  - [x] 1.1 创建目录结构 `src/models/`, `src/data/`, `src/training/`, `src/evaluation/`, `src/ensemble/`
  - [x] 1.2 创建 `trained_models/` 目录及各模型子目录
  - [x] 1.3 更新 `.gitignore` 排除数据文件和临时文件
  - _Requirements: 7.3_

- [ ] 2. 实现数据加载模块
  - [ ] 2.1 创建 `src/data/__init__.py` 和 `src/data/data_loader.py`
    - 实现 DRDataLoader 类
    - 支持从CSV加载数据
    - 实现图像预处理（resize, normalize）
    - 实现数据增强
  - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [ ] 2.2 编写数据加载器属性测试
    - **Property 2: Image Preprocessing Consistency**
    - **Property 3: Batch Size Consistency**
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [ ] 3. 实现模型基类
  - [ ] 3.1 创建 `src/models/__init__.py` 和 `src/models/base_model.py`
    - 实现 BaseModel 抽象基类
    - 定义 build(), compile(), train(), predict(), save(), load() 接口
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 3.2 编写模型基类属性测试
    - **Property 1: Model I/O Shape Consistency**
    - **Validates: Requirements 1.3, 1.4**

- [ ] 4. 实现训练模块
  - [ ] 4.1 创建 `src/training/__init__.py`, `src/training/trainer.py`, `src/training/callbacks.py`
    - 实现 ModelTrainer 类
    - 实现标准回调（EarlyStopping, ReduceLROnPlateau, ModelCheckpoint）
    - 实现训练历史保存
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 5. 实现评估模块
  - [ ] 5.1 创建 `src/evaluation/__init__.py`, `src/evaluation/evaluator.py`, `src/evaluation/metrics.py`
    - 实现 ModelEvaluator 类
    - 计算 accuracy, precision, recall, F1
    - 生成混淆矩阵
    - 保存评估报告
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ] 5.2 编写评估模块属性测试
    - **Property 6: Evaluation Metrics Consistency**
    - **Validates: Requirements 3.1, 3.3**

- [ ] 6. 创建训练脚本模板
  - [ ] 6.1 创建 `scripts/train_model.py`
    - 命令行参数支持选择模型
    - 加载数据、训练、保存模型
  - [ ] 6.2 创建 `scripts/evaluate_model.py`
    - 加载模型并评估
    - 生成评估报告
  - _Requirements: 2.4, 3.4_

- [ ] 7. Checkpoint - 框架验证
  - 确保所有基础模块可正常导入
  - 运行数据加载测试
  - 确认目录结构正确


## Phase 2: 模型实现（全员并行）

### 组长模型

- [ ] 8. 实现 ResNet-50 模型
  - [ ] 8.1 创建 `src/models/resnet50.py`
    - 继承 BaseModel
    - 使用 keras.applications.ResNet50 预训练权重
    - 添加分类头（GlobalAveragePooling + Dense）
  - [ ] 8.2 训练 ResNet-50 模型
    - 运行 `python scripts/train_model.py --model resnet50`
    - 保存到 `trained_models/resnet50/`
  - [ ] 8.3 评估 ResNet-50 模型
    - 运行 `python scripts/evaluate_model.py --model resnet50`
    - 生成评估报告
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

- [ ] 9. 实现 EfficientNet-B0 模型
  - [ ] 9.1 创建 `src/models/efficientnet_b0.py`
    - 继承 BaseModel
    - 使用 keras.applications.EfficientNetB0 预训练权重
  - [ ] 9.2 训练 EfficientNet-B0 模型
  - [ ] 9.3 评估 EfficientNet-B0 模型
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

### 成员A模型（前端）

- [ ] 10. 实现 VGG-16 模型
  - [ ] 10.1 创建 `src/models/vgg16.py`
    - 继承 BaseModel
    - 使用 keras.applications.VGG16 预训练权重
  - [ ] 10.2 训练 VGG-16 模型
  - [ ] 10.3 评估 VGG-16 模型
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

- [ ] 11. 实现 MobileNetV2 模型
  - [ ] 11.1 创建 `src/models/mobilenetv2.py`
    - 继承 BaseModel
    - 使用 keras.applications.MobileNetV2 预训练权重
  - [ ] 11.2 训练 MobileNetV2 模型
  - [ ] 11.3 评估 MobileNetV2 模型
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

### 成员B模型（代码强）

- [ ] 12. 实现 SE-ResNet 模型
  - [ ] 12.1 创建 `src/models/se_resnet.py`
    - 继承 BaseModel
    - 实现 SE (Squeeze-and-Excitation) 模块
    - 基于 ResNet 添加 SE 注意力机制
  - [ ] 12.2 训练 SE-ResNet 模型
  - [ ] 12.3 评估 SE-ResNet 模型
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

- [ ] 13. 实现 ResNeXt-50 模型
  - [ ] 13.1 创建 `src/models/resnext50.py`
    - 继承 BaseModel
    - 实现 ResNeXt 分组卷积结构
  - [ ] 13.2 训练 ResNeXt-50 模型
  - [ ] 13.3 评估 ResNeXt-50 模型
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

### 成员C模型

- [ ] 14. 实现 DenseNet-121 模型
  - [ ] 14.1 创建 `src/models/densenet121.py`
    - 继承 BaseModel
    - 使用 keras.applications.DenseNet121 预训练权重
  - [ ] 14.2 训练 DenseNet-121 模型
  - [ ] 14.3 评估 DenseNet-121 模型
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

- [ ] 15. 实现 InceptionV3 模型
  - [ ] 15.1 创建 `src/models/inceptionv3.py`
    - 继承 BaseModel
    - 使用 keras.applications.InceptionV3 预训练权重
    - 注意: InceptionV3 输入尺寸为 (299, 299, 3)
  - [ ] 15.2 训练 InceptionV3 模型
  - [ ] 15.3 评估 InceptionV3 模型
  - _Requirements: 1.1, 2.4, 3.4, 6.1_

- [ ] 16. Checkpoint - 模型训练验证
  - 确保所有8个模型训练完成
  - 确保所有模型文件保存在正确位置
  - 确保所有评估报告生成


## Phase 3: 集成学习（组长）

- [ ] 17. 实现集成学习模块
  - [ ] 17.1 创建 `src/ensemble/__init__.py`, `src/ensemble/ensemble_model.py`, `src/ensemble/voting.py`
    - 实现 EnsembleModel 类
    - 实现硬投票、软投票、加权投票策略
    - 实现模型加载功能
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ] 17.2 编写集成学习属性测试
    - **Property 4: Soft Voting Correctness**
    - **Property 5: Weighted Voting Correctness**
    - **Validates: Requirements 5.3, 5.4**

- [ ] 18. 创建集成学习脚本
  - [ ] 18.1 创建 `scripts/run_ensemble.py`
    - 加载所有训练好的模型
    - 运行集成预测
    - 生成对比报告
  - _Requirements: 5.5, 5.6_

- [ ] 19. 生成最终报告
  - [ ] 19.1 运行集成学习评估
    - 比较各模型单独性能
    - 比较不同投票策略性能
    - 生成最终对比报告
  - _Requirements: 5.6_

- [ ] 20. Checkpoint - 最终验证
  - 确保集成模型性能优于单个模型
  - 确保所有测试通过
  - 确保文档完整

## Phase 4: 文档和清理

- [ ] 21. 更新项目文档
  - [ ] 21.1 更新 README.md
    - 项目介绍
    - 安装说明
    - 使用方法
    - 模型性能对比
  - [ ] 21.2 更新 docs/项目整体规划.md
    - 完善团队分工
    - 添加最终结果
  - _Requirements: 7.1, 7.2_

## Notes

- 所有任务都必须完成，包括测试任务
- Phase 2 中各成员的任务可并行执行
- 每个成员完成后需创建 Pull Request 合并到 main 分支
- 模型训练时间较长，建议使用 GPU 加速
- InceptionV3 需要特殊处理输入尺寸 (299x299)
