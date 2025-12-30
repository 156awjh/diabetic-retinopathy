"""
几何变换数据增强模块 - 组员A
=============================

本模块实现几何变换相关的数据增强，包括：
1. 旋转 (Rotation)
2. 翻转 (Flip) - 水平/垂直
3. 裁剪 (Crop) - 随机裁剪

作者: 组员A
日期: 2024
"""

import cv2
import numpy as np
from typing import Tuple

try:
    from .base import BaseAugmentation
except ImportError:
    from base import BaseAugmentation


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
        """
        # 判断是否应用增强（根据概率）
        if np.random.random() >= self.probability:
            return image  # 不增强，直接返回原图
        
        # 1. 随机旋转
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = self._rotate(image, angle)
        
        # 2. 水平翻转（50%概率）
        if self.horizontal_flip and np.random.random() < 0.5:
            image = self._flip(image, horizontal=True)
        
        # 3. 垂直翻转（50%概率）
        if self.vertical_flip and np.random.random() < 0.5:
            image = self._flip(image, horizontal=False)
        
        # 4. 随机裁剪
        if self.crop_ratio[0] < 1.0:
            image = self._random_crop(image)
        
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
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 应用旋转，使用边界反射填充
        rotated = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )
        
        return rotated
    
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
        """
        if horizontal:
            return cv2.flip(image, 1)  # 1 = 水平翻转
        else:
            return cv2.flip(image, 0)  # 0 = 垂直翻转
    
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
        """
        h, w = image.shape[:2]
        
        # 随机选择裁剪比例
        crop_ratio = np.random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        
        # 计算裁剪尺寸
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        # 随机选择裁剪起点
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        # 裁剪
        cropped = image[top:top+new_h, left:left+new_w]
        
        # resize 回原始大小
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
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


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # 读取一张测试图片
    test_img_path = "data/data2/0/10_left.jpeg"
    
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        print(f"原始图像形状: {img.shape}")
        
        # 创建增强器
        aug = GeometricAugmentation(
            rotation_range=30,
            horizontal_flip=True,
            vertical_flip=False,
            crop_ratio=(0.8, 1.0),
            probability=1.0  # 测试时设为1，确保一定应用增强
        )
        
        # 应用增强
        augmented = aug(img)
        print(f"增强后图像形状: {augmented.shape}")
        
        # 保存结果
        output_dir = "output/augmentation_test"
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(f"{output_dir}/geometric_original.jpg", img)
        cv2.imwrite(f"{output_dir}/geometric_augmented.jpg", augmented)
        
        print(f"几何变换测试完成！结果保存在 {output_dir}/")
        print(f"配置: {aug.get_config()}")
    else:
        print(f"测试图片不存在: {test_img_path}")
        print("请确保数据集已下载到 data/data2/ 目录")
