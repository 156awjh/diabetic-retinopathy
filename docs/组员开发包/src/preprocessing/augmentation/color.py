"""
颜色变换数据增强模块 - 组员B负责实现
=====================================

本模块实现颜色变换相关的数据增强，包括：
1. 亮度调整 (Brightness)
2. 对比度调整 (Contrast)
3. 色调调整 (Hue)
4. 饱和度调整 (Saturation)

作者: [组员B姓名]
日期: 2024

使用示例:
---------
>>> from preprocessing.augmentation.color import ColorAugmentation
>>> 
>>> # 创建增强器
>>> aug = ColorAugmentation(
...     brightness_range=(0.8, 1.2),  # 亮度调整范围
...     contrast_range=(0.8, 1.2),    # 对比度调整范围
...     hue_range=(-0.1, 0.1),        # 色调调整范围
...     saturation_range=(0.8, 1.2),  # 饱和度调整范围
...     probability=0.5               # 应用概率
... )
>>> 
>>> # 应用增强
>>> augmented_image = aug(image)
"""

import numpy as np
from typing import Tuple, Optional
from .base import BaseAugmentation


class ColorAugmentation(BaseAugmentation):
    """
    颜色变换数据增强
    
    实现亮度、对比度、色调、饱和度等颜色变换。
    
    参数:
    -----
    brightness_range : tuple
        亮度调整范围，如 (0.8, 1.2) 表示 80%-120%
    contrast_range : tuple
        对比度调整范围
    hue_range : tuple
        色调调整范围，如 (-0.1, 0.1)
    saturation_range : tuple
        饱和度调整范围
    probability : float
        应用增强的概率
    
    TODO - 组员B需要实现:
    ---------------------
    1. __call__ 方法中的具体增强逻辑
    2. _adjust_brightness 方法 - 亮度调整
    3. _adjust_contrast 方法 - 对比度调整
    4. _adjust_hue 方法 - 色调调整
    5. _adjust_saturation 方法 - 饱和度调整
    
    提示:
    -----
    - 可以使用 OpenCV (cv2) 或 PIL 进行图像处理
    - 色调调整需要转换到 HSV 色彩空间
    - 注意数值范围裁剪 [0, 255]
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        probability: float = 0.5
    ):
        super().__init__(probability)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        应用颜色变换
        
        参数:
        -----
        image : np.ndarray
            输入图像，形状为 (H, W, C)，值范围 [0, 255]
        
        返回:
        -----
        np.ndarray
            增强后的图像
        
        TODO: 组员B实现此方法
        """
        # ============================================================
        # TODO: 组员B在此实现颜色变换逻辑
        # ============================================================
        # 
        # 参考实现步骤:
        # 1. 判断是否应用增强 (使用 self.probability)
        # 2. 随机调整亮度
        # 3. 随机调整对比度
        # 4. 随机调整色调
        # 5. 随机调整饱和度
        # 
        # 示例代码框架:
        # if np.random.random() < self.probability:
        #     # 亮度
        #     brightness_factor = np.random.uniform(*self.brightness_range)
        #     image = self._adjust_brightness(image, brightness_factor)
        #     
        #     # 对比度
        #     contrast_factor = np.random.uniform(*self.contrast_range)
        #     image = self._adjust_contrast(image, contrast_factor)
        #     
        #     # 色调
        #     hue_factor = np.random.uniform(*self.hue_range)
        #     image = self._adjust_hue(image, hue_factor)
        #     
        #     # 饱和度
        #     saturation_factor = np.random.uniform(*self.saturation_range)
        #     image = self._adjust_saturation(image, saturation_factor)
        # 
        # return image
        # ============================================================
        
        # 占位实现 - 直接返回原图
        print("[WARNING] ColorAugmentation 尚未实现，请组员B完成")
        return image
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整亮度
        
        参数:
        -----
        image : np.ndarray
            输入图像
        factor : float
            亮度因子，>1 变亮，<1 变暗
        
        返回:
        -----
        np.ndarray
            调整后的图像
        
        TODO: 组员B实现此方法
        提示: image * factor，注意裁剪到 [0, 255]
        """
        # TODO: 实现亮度调整逻辑
        return image
    
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整对比度
        
        参数:
        -----
        image : np.ndarray
            输入图像
        factor : float
            对比度因子
        
        返回:
        -----
        np.ndarray
            调整后的图像
        
        TODO: 组员B实现此方法
        提示: (image - mean) * factor + mean
        """
        # TODO: 实现对比度调整逻辑
        return image
    
    def _adjust_hue(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整色调
        
        参数:
        -----
        image : np.ndarray
            输入图像 (RGB)
        factor : float
            色调调整因子
        
        返回:
        -----
        np.ndarray
            调整后的图像
        
        TODO: 组员B实现此方法
        提示: 
        1. RGB -> HSV
        2. 调整 H 通道
        3. HSV -> RGB
        """
        # TODO: 实现色调调整逻辑
        return image
    
    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整饱和度
        
        参数:
        -----
        image : np.ndarray
            输入图像 (RGB)
        factor : float
            饱和度因子
        
        返回:
        -----
        np.ndarray
            调整后的图像
        
        TODO: 组员B实现此方法
        提示: 
        1. RGB -> HSV
        2. 调整 S 通道
        3. HSV -> RGB
        """
        # TODO: 实现饱和度调整逻辑
        return image
    
    def get_config(self) -> dict:
        """获取增强配置"""
        config = super().get_config()
        config.update({
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
            'hue_range': self.hue_range,
            'saturation_range': self.saturation_range
        })
        return config
