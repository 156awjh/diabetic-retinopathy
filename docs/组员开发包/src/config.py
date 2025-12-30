"""
项目配置文件
============

包含数据路径、模型参数、训练配置等全局设置。
"""

import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data" / "data2"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "output"
PREPROCESSING_OUTPUT_DIR = OUTPUT_DIR / "preprocessing"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"

# 确保输出目录存在
for dir_path in [OUTPUT_DIR, PREPROCESSING_OUTPUT_DIR, MODEL_OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 数据集配置
# ============================================================================

# 类别信息
CLASS_NAMES = {
    0: "No_DR",           # 无病变
    1: "Mild",            # 轻度
    2: "Moderate",        # 中度
    3: "Severe",          # 重度
    4: "Proliferative"    # 增殖性
}

NUM_CLASSES = 5

# 类别中文名称
CLASS_NAMES_CN = {
    0: "无病变",
    1: "轻度",
    2: "中度",
    3: "重度",
    4: "增殖性"
}

# ============================================================================
# 数据集划分配置
# ============================================================================

# 划分比例
TRAIN_RATIO = 0.70    # 训练集 70%
VAL_RATIO = 0.15      # 验证集 15%
TEST_RATIO = 0.15     # 测试集 15%

# 随机种子（确保可复现）
RANDOM_SEED = 42

# ============================================================================
# 类别平衡配置
# ============================================================================

# 过采样策略: 'median' 或 'max'
OVERSAMPLE_STRATEGY = 'median'

# 类别权重策略: 'balanced' 或 'sqrt_balanced'
CLASS_WEIGHT_STRATEGY = 'balanced'

# ============================================================================
# 图像配置
# ============================================================================

# 图像尺寸（模型输入）
IMAGE_SIZE = (224, 224)

# 图像通道数
IMAGE_CHANNELS = 3

# 支持的图像格式
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# ============================================================================
# 训练配置（供后续使用）
# ============================================================================

# 批次大小
BATCH_SIZE = 32

# 训练轮数
EPOCHS = 50

# 初始学习率
LEARNING_RATE = 0.001

# 早停耐心值
EARLY_STOPPING_PATIENCE = 5

# ============================================================================
# 输出文件名
# ============================================================================

# 数据集划分文件
SPLIT_CSV_FILE = PREPROCESSING_OUTPUT_DIR / "dataset_splits.csv"
TRAIN_CSV_FILE = PREPROCESSING_OUTPUT_DIR / "train.csv"
VAL_CSV_FILE = PREPROCESSING_OUTPUT_DIR / "val.csv"
TEST_CSV_FILE = PREPROCESSING_OUTPUT_DIR / "test.csv"

# 过采样后的训练集
OVERSAMPLED_TRAIN_CSV = PREPROCESSING_OUTPUT_DIR / "train_oversampled.csv"

# 类别权重文件
CLASS_WEIGHTS_FILE = PREPROCESSING_OUTPUT_DIR / "class_weights.json"

# 可视化图片
DISTRIBUTION_PLOT = PREPROCESSING_OUTPUT_DIR / "class_distribution.png"
BALANCE_COMPARISON_PLOT = PREPROCESSING_OUTPUT_DIR / "balance_comparison.png"
