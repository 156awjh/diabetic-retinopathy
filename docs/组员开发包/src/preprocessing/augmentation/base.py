"""
数据增强基类
============

所有数据增强类都应继承此基类，并实现 __call__ 方法。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseAugmentation(ABC):
    """
    数据增强基类
    
    所有组员实现的数据增强类都应继承此类。
    
    使用示例:
    ---------
    class MyAugmentation(BaseAugmentation):
        def __init__(self, probability=0.5):
            super().__init__(probability)
            # 初始化你的参数
        
        def __call__(self, image):
            if np.random.random() < self.probability:
                # 应用你的增强
                image = your_augmentation(image)
            return image
    
    属性:
    -----
    probability : float
        应用增强的概率，范围 [0, 1]
    """
    
    def __init__(self, probability: float = 0.5):
        """
        初始化基类
        
        参数:
        -----
        probability : float
            应用增强的概率，默认 0.5
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"probability 必须在 [0, 1] 范围内，当前值: {probability}")
        self.probability = probability
    
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        
        参数:
        -----
        image : np.ndarray
            输入图像，形状为 (H, W, C) 或 (H, W)
        
        返回:
        -----
        np.ndarray
            增强后的图像
        
        注意:
        -----
        子类必须实现此方法！
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取增强配置
        
        返回:
        -----
        dict
            包含增强参数的字典
        """
        return {
            'name': self.__class__.__name__,
            'probability': self.probability
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(probability={self.probability})"


class AugmentationPipeline:
    """
    数据增强管道
    
    将多个增强操作组合成一个管道，按顺序应用。
    
    使用示例:
    ---------
    pipeline = AugmentationPipeline([
        GeometricAugmentation(probability=0.5),
        ColorAugmentation(probability=0.3),
    ])
    augmented_image = pipeline(image)
    """
    
    def __init__(self, augmentations: list):
        """
        初始化增强管道
        
        参数:
        -----
        augmentations : list
            BaseAugmentation 实例列表
        """
        self.augmentations = augmentations
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        按顺序应用所有增强
        
        参数:
        -----
        image : np.ndarray
            输入图像
        
        返回:
        -----
        np.ndarray
            增强后的图像
        """
        for aug in self.augmentations:
            image = aug(image)
        return image
    
    def get_config(self) -> Dict[str, Any]:
        """获取管道配置"""
        return {
            'augmentations': [aug.get_config() for aug in self.augmentations]
        }
