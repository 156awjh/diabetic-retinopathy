"""
几何变换数据增强模块 - 组员A负责实现
=====================================

本模块实现几何变换相关的数据增强，包括：
1. 旋转 (Rotation)
2. 翻转 (Flip) - 水平/垂直
3. 裁剪 (Crop) - 随机裁剪/中心裁剪

作者: [组员A姓名]
日期: 2024

使用示例:
---------
>>> from preprocessing.augmentation.geometric import GeometricAugmentation
>>> 
>>> # 创建增强器
>>> aug = GeometricAugmentation(
...     rotation_range=30,      # 旋转角度范围 [-30, 30]
...     horizontal_flip=True,   # 启用水平翻转
...     vertical_flip=False,    # 禁用垂直翻转
...     crop_ratio=(0.8, 1.0),  # 随机裁剪比例
...     probability=0.5         # 应用概率
... )
>>> 
>>> # 应用增强
>>> augmented_image = aug(image)
"""

import numpy as np
from typing import Tuple, Optional
from .base import BaseAugmentation


class GeometricAugmentation(BaseAugmentation):
    """
    几何变换数据增强
    
    实现旋转、翻转、裁剪等几何变换。
    
    参数:
    -----
    rotation_range : float
        旋转角度范围，如 30 表示 [-30°, 30°] 随机旋转
    horizontal_flip : bool
        是否启用水平翻转
    vertical_flip : bool
        是否启用垂直翻转
    crop_ratio : tuple
        随机裁剪比例范围，如 (0.8, 1.0) 表示裁剪 80%-100% 的区域
    probability : float
        应用增强的概率
    
    TODO - 组员A需要实现:
    ---------------------
    1. __call__ 方法中的具体增强逻辑
    2. _rotate 方法 - 图像旋转
    3. _flip 方法 - 图像翻转
    4. _random_crop 方法 - 随机裁剪
    
    提示:
    -----
    - 可以使用 OpenCV (cv2) 或 PIL 进行图像处理
    - 旋转时注意保持图像尺寸
    - 裁剪后需要 resize 回原始尺寸
    """
    
    def __init__(
        self,
        rotation_range: float = 30,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        crop_ratio: Tuple[float, float] = (0.8, 1.0),
        probability: float = 0.5
    ):
        super().__init__(probability)
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.crop_ratio = crop_ratio
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        应用几何变换
        
        参数:
        -----
        image : np.ndarray
            输入图像，形状为 (H, W, C)
        
        返回:
        -----
        np.ndarray
            增强后的图像
        
        TODO: 组员A实现此方法
        """
        # ============================================================
        # TODO: 组员A在此实现几何变换逻辑
        # ============================================================
        # 
        # 参考实现步骤:
        # 1. 判断是否应用增强 (使用 self.probability)
        # 2. 随机旋转
        # 3. 随机翻转
        # 4. 随机裁剪
        # 
        # 示例代码框架:
        # if np.random.random() < self.probability:
        #     # 旋转
        #     if self.rotation_range > 0:
        #         angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        #         image = self._rotate(image, angle)
        #     
        #     # 水平翻转
        #     if self.horizontal_flip and np.random.random() < 0.5:
        #         image = self._flip(image, horizontal=True)
        #     
        #     # 垂直翻转
        #     if self.vertical_flip and np.random.random() < 0.5:
        #         image = self._flip(image, horizontal=False)
        #     
        #     # 随机裁剪
        #     if self.crop_ratio[0] < 1.0:
        #         image = self._random_crop(image)
        # 
        # return image
        # ============================================================
        
        # 占位实现 - 直接返回原图
        print("[WARNING] GeometricAugmentation 尚未实现，请组员A完成")
        return image
    
    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像
        
        参数:
        -----
        image : np.ndarray
            输入图像
        angle : float
            旋转角度（度）
        
        返回:
        -----
        np.ndarray
            旋转后的图像
        
        TODO: 组员A实现此方法
        提示: 使用 cv2.getRotationMatrix2D 和 cv2.warpAffine
        """
        # TODO: 实现旋转逻辑
        return image
    
    def _flip(self, image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """
        翻转图像
        
        参数:
        -----
        image : np.ndarray
            输入图像
        horizontal : bool
            True 为水平翻转，False 为垂直翻转
        
        返回:
        -----
        np.ndarray
            翻转后的图像
        
        TODO: 组员A实现此方法
        提示: 使用 cv2.flip 或 np.flip
        """
        # TODO: 实现翻转逻辑
        return image
    
    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        """
        随机裁剪图像
        
        参数:
        -----
        image : np.ndarray
            输入图像
        
        返回:
        -----
        np.ndarray
            裁剪后的图像（resize 回原始尺寸）
        
        TODO: 组员A实现此方法
        提示: 
        1. 随机选择裁剪比例
        2. 随机选择裁剪位置
        3. 裁剪后 resize 回原始尺寸
        """
        # TODO: 实现随机裁剪逻辑
        return image
    
    def get_config(self) -> dict:
        """获取增强配置"""
        config = super().get_config()
        config.update({
            'rotation_range': self.rotation_range,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'crop_ratio': self.crop_ratio
        })
        return config
