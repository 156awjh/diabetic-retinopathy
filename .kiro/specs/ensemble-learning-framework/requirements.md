# Requirements Document

## Introduction

本项目是一个糖尿病视网膜病变（Diabetic Retinopathy）图像分类系统，采用集成学习方法，整合8个深度学习模型进行5分类任务。项目由4名团队成员协作完成，每人负责训练2个模型，最终通过集成学习策略提升分类准确率。

## Glossary

- **Ensemble_System**: 集成学习系统，负责整合多个模型的预测结果
- **Base_Model**: 基础模型类，所有深度学习模型的父类
- **Model_Trainer**: 模型训练器，负责模型的训练流程
- **Model_Evaluator**: 模型评估器，负责生成评估报告
- **Data_Loader**: 数据加载器，负责加载和预处理图像数据
- **Voting_Strategy**: 投票策略，用于集成多个模型的预测结果
- **DR_Class**: 糖尿病视网膜病变分类（0-无病变, 1-轻度, 2-中度, 3-重度, 4-增殖性）

## Requirements

### Requirement 1: 统一模型接口

**User Story:** As a 团队成员, I want 所有模型遵循统一的接口规范, so that 我们可以方便地进行模型集成和评估。

#### Acceptance Criteria

1. THE Base_Model SHALL define a standard interface including `build()`, `compile()`, `train()`, `predict()`, and `save()` methods
2. WHEN a team member creates a new model, THE Base_Model SHALL provide default implementations for common operations
3. THE Base_Model SHALL accept consistent input shape of (224, 224, 3) for all models
4. THE Base_Model SHALL output predictions with shape (batch_size, 5) representing 5 DR classes

### Requirement 2: 模型训练流程

**User Story:** As a 团队成员, I want 一个标准化的训练流程, so that 我可以在自己的电脑上独立训练模型并保存结果。

#### Acceptance Criteria

1. WHEN training starts, THE Model_Trainer SHALL load data from the preprocessed CSV files (train.csv, val.csv)
2. WHEN training starts, THE Model_Trainer SHALL apply data augmentation to training images
3. WHILE training, THE Model_Trainer SHALL use callbacks for early stopping, learning rate reduction, and model checkpointing
4. WHEN training completes, THE Model_Trainer SHALL save the model in `.keras` format with a standardized naming convention
5. WHEN training completes, THE Model_Trainer SHALL save training history (loss, accuracy curves) as JSON and PNG files
6. IF training is interrupted, THEN THE Model_Trainer SHALL support resuming from the last checkpoint

### Requirement 3: 模型评估

**User Story:** As a 团队成员, I want 统一的评估标准, so that 我们可以公平地比较各个模型的性能。

#### Acceptance Criteria

1. WHEN evaluating a model, THE Model_Evaluator SHALL compute accuracy, precision, recall, F1-score for each class
2. WHEN evaluating a model, THE Model_Evaluator SHALL generate a confusion matrix visualization
3. WHEN evaluating a model, THE Model_Evaluator SHALL compute macro and weighted average metrics
4. WHEN evaluation completes, THE Model_Evaluator SHALL save results to a standardized JSON report file
5. THE Model_Evaluator SHALL use the same test set (test.csv) for all model evaluations

### Requirement 4: 数据加载

**User Story:** As a 团队成员, I want 一个统一的数据加载器, so that 所有模型使用相同的数据预处理流程。

#### Acceptance Criteria

1. WHEN loading images, THE Data_Loader SHALL resize images to (224, 224) pixels
2. WHEN loading images, THE Data_Loader SHALL normalize pixel values to [0, 1] range
3. THE Data_Loader SHALL support batch loading with configurable batch size
4. THE Data_Loader SHALL apply class weights during training to handle class imbalance
5. WHEN loading training data, THE Data_Loader SHALL optionally apply data augmentation

### Requirement 5: 集成学习

**User Story:** As a 组长, I want 将所有训练好的模型集成起来, so that 我们可以获得比单个模型更好的分类性能。

#### Acceptance Criteria

1. WHEN loading models for ensemble, THE Ensemble_System SHALL load all 8 trained model files from the `trained_models/` directory
2. THE Ensemble_System SHALL support multiple voting strategies: hard voting, soft voting, and weighted voting
3. WHEN using soft voting, THE Ensemble_System SHALL average the probability outputs from all models
4. WHEN using weighted voting, THE Ensemble_System SHALL allow configuring weights based on individual model performance
5. WHEN ensemble prediction completes, THE Ensemble_System SHALL output the final class prediction and confidence scores
6. THE Ensemble_System SHALL generate a comparison report showing individual model vs ensemble performance

### Requirement 6: 模型保存规范

**User Story:** As a 团队成员, I want 统一的模型保存格式, so that 组长可以方便地加载和集成所有模型。

#### Acceptance Criteria

1. WHEN saving a model, THE Model_Trainer SHALL use the naming format: `{model_name}_best.keras`
2. WHEN saving a model, THE Model_Trainer SHALL also save model metadata including: training date, final metrics, hyperparameters
3. THE trained model files SHALL be saved in the `trained_models/{model_name}/` directory structure
4. WHEN saving training artifacts, THE Model_Trainer SHALL include: model weights, training history, evaluation report

### Requirement 7: Git协作流程

**User Story:** As a 团队成员, I want 清晰的Git分支策略, so that 我们可以并行开发而不产生冲突。

#### Acceptance Criteria

1. THE project SHALL use feature branches named `model-{model_names}` for each team member
2. WHEN a team member completes model training, THE team member SHALL create a pull request to merge into main branch
3. THE `.gitignore` SHALL exclude large data files and temporary outputs
4. THE `trained_models/` directory SHALL be tracked in Git for model sharing

### Requirement 8: 配置管理

**User Story:** As a 团队成员, I want 集中管理的配置文件, so that 所有人使用一致的超参数和路径设置。

#### Acceptance Criteria

1. THE config.py SHALL define all shared hyperparameters: batch_size, epochs, learning_rate, image_size
2. THE config.py SHALL define all file paths relative to project root
3. WHEN a team member needs custom settings, THE config.py SHALL support environment-specific overrides
4. THE config.py SHALL define model-specific configurations for each of the 8 models
