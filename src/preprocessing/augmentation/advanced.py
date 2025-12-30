"""
高级数据增强模块 - 组员C
=========================

本模块实现高级数据增强技术：
1. Mixup - 混合两张图像及其标签
2. CutMix - 剪切并粘贴图像区域

作者: 组员C
日期: 2024
"""

import cv2
import numpy as np
from typing import Tuple

try:
    from .base import BaseAugmentation
except ImportError:
    from base import BaseAugmentation


class MixupAugmentation(BaseAugmentation):
    """
    Mixup 数据增强
    
    将两张图像按比例混合：
    mixed = λ * image1 + (1-λ) * image2
    
    参数:
    -----
    alpha : float
        Beta 分布参数，控制混合程度。
        alpha 越小，混合越极端（接近 0 或 1）
        alpha 越大，混合越均匀（接近 0.5）
    probability : float
        应用增强的概率
    
    参考论文:
    ---------
    mixup: Beyond Empirical Risk Minimization
    https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        super().__init__(probability)
        self.alpha = alpha
    
    def __call__(
        self,
        image1: np.ndarray,
        label1: int,
        image2: np.ndarray,
        label2: int,
        num_classes: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用 Mixup 增强
        
        参数:
        -----
        image1 : np.ndarray
            第一张图像
        label1 : int
            第一张图像的标签
        image2 : np.ndarray
            第二张图像
        label2 : int
            第二张图像的标签
        num_classes : int
            类别数量，默认 5
        
        返回:
        -----
        tuple
            (mixed_image, mixed_label)
            - mixed_image: 混合后的图像
            - mixed_label: 混合后的标签（软标签，one-hot 形式）
        """
        # 判断是否应用增强
        if np.random.random() >= self.probability:
            # 不增强，返回原图和 one-hot 标签
            one_hot = np.zeros(num_classes)
            one_hot[label1] = 1.0
            return image1, one_hot
        
        # 确保两张图片大小相同
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 从 Beta 分布采样混合比例 λ
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合图像
        mixed_image = lam * image1.astype(np.float32) + (1 - lam) * image2.astype(np.float32)
        mixed_image = mixed_image.astype(np.uint8)
        
        # 混合标签（软标签）
        mixed_label = np.zeros(num_classes)
        mixed_label[label1] = lam
        mixed_label[label2] += (1 - lam)  # 用 += 因为 label1 可能等于 label2
        
        return mixed_image, mixed_label
    
    def get_config(self) -> dict:
        """获取增强配置"""
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config


class CutMixAugmentation(BaseAugmentation):
    """
    CutMix 数据增强
    
    从一张图像中剪切矩形区域，粘贴到另一张图像上。
    
    参数:
    -----
    alpha : float
        Beta 分布参数，控制剪切区域大小
    probability : float
        应用增强的概率
    
    参考论文:
    ---------
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    https://arxiv.org/abs/1905.04899
    """
    
    def __init__(self, alpha: float = 1.0, probability: float = 0.5):
        super().__init__(probability)
        self.alpha = alpha
    
    def __call__(
        self,
        image1: np.ndarray,
        label1: int,
        image2: np.ndarray,
        label2: int,
        num_classes: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用 CutMix 增强
        
        参数:
        -----
        image1 : np.ndarray
            第一张图像（基础图像）
        label1 : int
            第一张图像的标签
        image2 : np.ndarray
            第二张图像（剪切来源）
        label2 : int
            第二张图像的标签
        num_classes : int
            类别数量，默认 5
        
        返回:
        -----
        tuple
            (mixed_image, mixed_label)
            - mixed_image: 混合后的图像
            - mixed_label: 混合后的标签（按面积比例）
        """
        # 判断是否应用增强
        if np.random.random() >= self.probability:
            one_hot = np.zeros(num_classes)
            one_hot[label1] = 1.0
            return image1, one_hot
        
        # 确保两张图片大小相同
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 采样 λ
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 获取图像尺寸
        H, W = image1.shape[:2]
        
        # 获取随机矩形框
        x1, y1, x2, y2 = self._get_rand_bbox(W, H, lam)
        
        # 剪切粘贴
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        
        # 计算实际混合比例（按面积）
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        
        # 混合标签
        mixed_label = np.zeros(num_classes)
        mixed_label[label1] = lam
        mixed_label[label2] += (1 - lam)
        
        return mixed_image, mixed_label
    
    def _get_rand_bbox(
        self,
        width: int,
        height: int,
        lam: float
    ) -> Tuple[int, int, int, int]:
        """
        生成随机矩形框
        
        参数:
        -----
        width : int
            图像宽度
        height : int
            图像高度
        lam : float
            混合比例，决定剪切区域大小
        
        返回:
        -----
        tuple
            (x1, y1, x2, y2) 矩形框坐标
        """
        # 计算剪切区域大小
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        
        # 随机选择剪切中心
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        
        # 计算边界（确保不超出图像范围）
        x1 = np.clip(cx - cut_w // 2, 0, width)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        
        return x1, y1, x2, y2
    
    def get_config(self) -> dict:
        """获取增强配置"""
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # 读取两张测试图片
    test_img_path1 = "data/data2/0/10_left.jpeg"
    test_img_path2 = "data/data2/2/1002_right.jpeg"
    
    if os.path.exists(test_img_path1) and os.path.exists(test_img_path2):
        img1 = cv2.imread(test_img_path1)
        img2 = cv2.imread(test_img_path2)
        
        # 确保两张图片大小相同
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        print(f"图像1形状: {img1.shape}, 标签: 0")
        print(f"图像2形状: {img2.shape}, 标签: 2")
        
        # 创建输出目录
        output_dir = "output/augmentation_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试 Mixup
        print("\n--- 测试 Mixup ---")
        mixup = MixupAugmentation(alpha=0.2, probability=1.0)
        mixed_img, mixed_label = mixup(img1, 0, img2, 2)
        print(f"混合标签: {mixed_label}")
        print(f"标签解释: 类别0占{mixed_label[0]:.2%}, 类别2占{mixed_label[2]:.2%}")
        
        cv2.imwrite(f"{output_dir}/mixup_img1.jpg", img1)
        cv2.imwrite(f"{output_dir}/mixup_img2.jpg", img2)
        cv2.imwrite(f"{output_dir}/mixup_result.jpg", mixed_img)
        
        # 测试 CutMix
        print("\n--- 测试 CutMix ---")
        cutmix = CutMixAugmentation(alpha=1.0, probability=1.0)
        cutmix_img, cutmix_label = cutmix(img1, 0, img2, 2)
        print(f"混合标签: {cutmix_label}")
        print(f"标签解释: 类别0占{cutmix_label[0]:.2%}, 类别2占{cutmix_label[2]:.2%}")
        
        cv2.imwrite(f"{output_dir}/cutmix_result.jpg", cutmix_img)
        
        print(f"\n高级增强测试完成！结果保存在 {output_dir}/")
    else:
        print(f"测试图片不存在")
        print("请确保数据集已下载到 data/data2/ 目录")
