"""
颜色变换数据增强模块 - 组员B
=============================

本模块实现颜色变换相关的数据增强，包括：
1. 亮度调整 (Brightness)
2. 对比度调整 (Contrast)
3. 色调调整 (Hue)
4. 饱和度调整 (Saturation)

作者: 组员B
日期: 2024
"""

import cv2
import numpy as np
from typing import Tuple

try:
    from .base import BaseAugmentation
except ImportError:
    from base import BaseAugmentation


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
        """
        # 判断是否应用增强
        if np.random.random() >= self.probability:
            return image
        
        # 1. 调整亮度
        brightness_factor = np.random.uniform(*self.brightness_range)
        image = self._adjust_brightness(image, brightness_factor)
        
        # 2. 调整对比度
        contrast_factor = np.random.uniform(*self.contrast_range)
        image = self._adjust_contrast(image, contrast_factor)
        
        # 3. 调整色调
        hue_factor = np.random.uniform(*self.hue_range)
        image = self._adjust_hue(image, hue_factor)
        
        # 4. 调整饱和度
        saturation_factor = np.random.uniform(*self.saturation_range)
        image = self._adjust_saturation(image, saturation_factor)
        
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
        """
        # 将图片转为浮点数
        img_float = image.astype(np.float32)
        
        # 乘以亮度因子
        img_float = img_float * factor
        
        # 裁剪到 [0, 255] 范围
        img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)
    
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
        """
        img_float = image.astype(np.float32)
        
        # 计算图片均值
        mean = np.mean(img_float)
        
        # 对比度调整公式: (pixel - mean) * factor + mean
        img_float = (img_float - mean) * factor + mean
        
        # 裁剪到 [0, 255]
        img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)
    
    def _adjust_hue(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整色调
        
        参数:
        -----
        image : np.ndarray
            输入图像 (BGR)
        factor : float
            色调调整因子
        
        返回:
        -----
        np.ndarray
            调整后的图像
        """
        # BGR 转 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # H 通道范围是 [0, 180]，调整色调
        hsv[:, :, 0] = hsv[:, :, 0] + factor * 180
        
        # 确保 H 在有效范围内（循环）
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
        
        # HSV 转回 BGR
        hsv = hsv.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return bgr
    
    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整饱和度
        
        参数:
        -----
        image : np.ndarray
            输入图像 (BGR)
        factor : float
            饱和度因子
        
        返回:
        -----
        np.ndarray
            调整后的图像
        """
        # BGR 转 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # S 通道乘以因子
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # HSV 转回 BGR
        hsv = hsv.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return bgr
    
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
        aug = ColorAugmentation(
            brightness_range=(0.7, 1.3),
            contrast_range=(0.8, 1.2),
            hue_range=(-0.1, 0.1),
            saturation_range=(0.8, 1.2),
            probability=1.0  # 测试时设为1，确保一定应用增强
        )
        
        # 应用增强
        augmented = aug(img)
        print(f"增强后图像形状: {augmented.shape}")
        
        # 保存结果
        output_dir = "output/augmentation_test"
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(f"{output_dir}/color_original.jpg", img)
        cv2.imwrite(f"{output_dir}/color_augmented.jpg", augmented)
        
        print(f"颜色变换测试完成！结果保存在 {output_dir}/")
        print(f"配置: {aug.get_config()}")
    else:
        print(f"测试图片不存在: {test_img_path}")
        print("请确保数据集已下载到 data/data2/ 目录")
