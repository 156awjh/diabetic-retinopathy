"""
糖尿病视网膜病变数据加载器
==========================

提供统一的数据加载接口，支持训练、验证、测试数据集的加载。
所有团队成员使用此加载器确保数据处理一致性。

使用示例:
---------
>>> loader = DRDataLoader(batch_size=32)
>>> train_ds = loader.load_train_data(augment=True)
>>> val_ds = loader.load_val_data()
>>> test_ds = loader.load_test_data()
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PREPROCESSING_OUTPUT_DIR, TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE,
    IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES
)


class DRDataLoader:
    """
    糖尿病视网膜病变数据加载器
    
    属性:
    -----
    image_size : tuple
        图像尺寸，默认 (224, 224)
    batch_size : int
        批次大小，默认 32
    num_classes : int
        类别数量，默认 5
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = None,
                 batch_size: int = None,
                 preprocessing_dir: str = None):
        """
        初始化数据加载器
        
        参数:
        -----
        image_size : tuple, optional
            图像尺寸，默认使用配置文件中的 IMAGE_SIZE
        batch_size : int, optional
            批次大小，默认使用配置文件中的 BATCH_SIZE
        preprocessing_dir : str, optional
            预处理输出目录，默认使用配置文件中的 PREPROCESSING_OUTPUT_DIR
        """
        self.image_size = image_size or IMAGE_SIZE
        self.batch_size = batch_size or BATCH_SIZE
        self.preprocessing_dir = Path(preprocessing_dir) if preprocessing_dir else PREPROCESSING_OUTPUT_DIR
        self.num_classes = NUM_CLASSES
    
    def _load_and_preprocess(self, filepath: tf.Tensor, label: tf.Tensor):
        """
        加载并预处理单张图片
        
        参数:
        -----
        filepath : tf.Tensor
            图片路径
        label : tf.Tensor
            标签
        
        返回:
        -----
        tuple
            (预处理后的图像, one-hot编码的标签)
        """
        # 读取图片
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # 调整大小
        image = tf.image.resize(image, self.image_size)
        
        # 归一化到 [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # One-hot 编码标签
        label = tf.one_hot(label, self.num_classes)
        
        return image, label

    def _augment(self, image: tf.Tensor, label: tf.Tensor):
        """
        数据增强
        
        参数:
        -----
        image : tf.Tensor
            输入图像
        label : tf.Tensor
            标签
        
        返回:
        -----
        tuple
            (增强后的图像, 标签)
        """
        # 随机水平翻转
        image = tf.image.random_flip_left_right(image)
        
        # 随机垂直翻转
        image = tf.image.random_flip_up_down(image)
        
        # 随机亮度调整
        image = tf.image.random_brightness(image, max_delta=0.2)
        
        # 随机对比度调整
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # 随机饱和度调整
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        
        # 确保像素值在 [0, 1] 范围内
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def _create_dataset(self, csv_path: str, 
                        augment: bool = False,
                        shuffle: bool = False) -> tf.data.Dataset:
        """
        从CSV创建数据集
        
        参数:
        -----
        csv_path : str
            CSV文件路径
        augment : bool
            是否应用数据增强
        shuffle : bool
            是否打乱数据
        
        返回:
        -----
        tf.data.Dataset
            TensorFlow数据集
        """
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        filepaths = df['filepath'].values
        labels = df['label'].values.astype(np.int32)
        
        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        
        # 加载和预处理
        dataset = dataset.map(
            self._load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 打乱数据
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # 数据增强
        if augment:
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # 批处理
        dataset = dataset.batch(self.batch_size)
        
        # 预取
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def load_train_data(self, augment: bool = True) -> tf.data.Dataset:
        """
        加载训练数据
        
        参数:
        -----
        augment : bool
            是否应用数据增强，默认 True
        
        返回:
        -----
        tf.data.Dataset
            训练数据集
        """
        return self._create_dataset(
            str(TRAIN_CSV_FILE),
            augment=augment,
            shuffle=True
        )
    
    def load_val_data(self) -> tf.data.Dataset:
        """
        加载验证数据
        
        返回:
        -----
        tf.data.Dataset
            验证数据集
        """
        return self._create_dataset(
            str(VAL_CSV_FILE),
            augment=False,
            shuffle=False
        )
    
    def load_test_data(self) -> tf.data.Dataset:
        """
        加载测试数据
        
        返回:
        -----
        tf.data.Dataset
            测试数据集
        """
        return self._create_dataset(
            str(TEST_CSV_FILE),
            augment=False,
            shuffle=False
        )
    
    def get_class_weights(self) -> dict:
        """
        计算类别权重（用于处理类别不平衡）
        
        返回:
        -----
        dict
            类别权重字典 {class_index: weight}
        """
        df = pd.read_csv(TRAIN_CSV_FILE)
        class_counts = df['label'].value_counts().sort_index()
        total = len(df)
        
        weights = {}
        for class_idx, count in class_counts.items():
            weights[class_idx] = total / (self.num_classes * count)
        
        return weights


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试数据加载器...")
    
    loader = DRDataLoader(batch_size=4)
    
    # 测试训练数据加载
    print("\n加载训练数据...")
    train_ds = loader.load_train_data(augment=True)
    
    for images, labels in train_ds.take(1):
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"像素值范围: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
    
    # 测试类别权重
    print("\n类别权重:")
    weights = loader.get_class_weights()
    for cls, weight in weights.items():
        print(f"  类别 {cls}: {weight:.4f}")
    
    print("\n数据加载器测试完成！")
