"""
高级数据增强模块 - 组员C负责实现
=================================

本模块实现高级数据增强技术，二选一实现：
1. Mixup - 混合两张图像及其标签
2. CutMix - 剪切并粘贴图像区域

作者: [组员C姓名]
日期: 2024

使用示例:
---------
>>> from preprocessing.augmentation.advanced import MixupAugmentation, CutMixAugmentation
>>> 
>>> # Mixup 示例
>>> mixup = MixupAugmentation(alpha=0.2)
>>> mixed_image, mixed_label = mixup(image1, label1, image2, label2)
>>> 
>>> # CutMix 示例
>>> cutmix = CutMixAugmentation(alpha=1.0)
>>> mixed_image, mixed_label = cutmix(image1, label1, image2, label2)

技术说明:
---------
Mixup:
    - 将两张图像按比例混合: mixed = λ * image1 + (1-λ) * image2
    - 标签也按相同比例混合: mixed_label = λ * label1 + (1-λ) * label2
    - λ 从 Beta(α, α) 分布中采样

CutMix:
    - 从 image2 中剪切一个矩形区域，粘贴到 image1 上
    - 标签按面积比例混合
    - 矩形大小由 λ 决定，λ 从 Beta(α, α) 分布中采样
"""

import numpy as np
from typing import Tuple, Optional
from .base import BaseAugmentation


class MixupAugmentation(BaseAugmentation):
    """
    Mixup 数据增强
    
    将两张图像及其标签按比例混合。
    
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
    
    TODO - 组员C需要实现:
    ---------------------
    1. __call__ 方法中的 Mixup 逻辑
    2. 正确处理标签混合（用于损失函数计算）
    """
    
    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        super().__init__(probability)
        self.alpha = alpha
    
    def __call__(
        self,
        image1: np.ndarray,
        label1: int,
        image2: np.ndarray,
        label2: int
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
        
        返回:
        -----
        tuple
            (mixed_image, mixed_label)
            - mixed_image: 混合后的图像
            - mixed_label: 混合后的标签（one-hot 或软标签）
        
        TODO: 组员C实现此方法
        """
        # ============================================================
        # TODO: 组员C在此实现 Mixup 逻辑
        # ============================================================
        # 
        # 参考实现步骤:
        # 1. 判断是否应用增强 (使用 self.probability)
        # 2. 从 Beta(alpha, alpha) 分布采样 λ
        # 3. 混合图像: mixed = λ * image1 + (1-λ) * image2
        # 4. 混合标签: 返回 (label1, label2, λ) 或 one-hot 软标签
        # 
        # 示例代码框架:
        # if np.random.random() < self.probability:
        #     # 采样混合比例
        #     lam = np.random.beta(self.alpha, self.alpha)
        #     
        #     # 混合图像
        #     mixed_image = lam * image1 + (1 - lam) * image2
        #     mixed_image = mixed_image.astype(np.uint8)
        #     
        #     # 混合标签 (返回软标签)
        #     # 方式1: 返回元组 (label1, label2, lam)
        #     # 方式2: 返回 one-hot 软标签
        #     num_classes = 5
        #     mixed_label = np.zeros(num_classes)
        #     mixed_label[label1] = lam
        #     mixed_label[label2] += (1 - lam)
        #     
        #     return mixed_image, mixed_label
        # else:
        #     # 不应用增强，返回原图和 one-hot 标签
        #     num_classes = 5
        #     one_hot = np.zeros(num_classes)
        #     one_hot[label1] = 1.0
        #     return image1, one_hot
        # ============================================================
        
        # 占位实现 - 直接返回原图
        print("[WARNING] MixupAugmentation 尚未实现，请组员C完成")
        num_classes = 5
        one_hot = np.zeros(num_classes)
        one_hot[label1] = 1.0
        return image1, one_hot
    
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
    
    TODO - 组员C需要实现:
    ---------------------
    1. __call__ 方法中的 CutMix 逻辑
    2. _get_rand_bbox 方法 - 生成随机矩形框
    """
    
    def __init__(self, alpha: float = 1.0, probability: float = 0.5):
        super().__init__(probability)
        self.alpha = alpha
    
    def __call__(
        self,
        image1: np.ndarray,
        label1: int,
        image2: np.ndarray,
        label2: int
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
        
        返回:
        -----
        tuple
            (mixed_image, mixed_label)
            - mixed_image: 混合后的图像
            - mixed_label: 混合后的标签（按面积比例）
        
        TODO: 组员C实现此方法
        """
        # ============================================================
        # TODO: 组员C在此实现 CutMix 逻辑
        # ============================================================
        # 
        # 参考实现步骤:
        # 1. 判断是否应用增强 (使用 self.probability)
        # 2. 从 Beta(alpha, alpha) 分布采样 λ
        # 3. 根据 λ 计算剪切区域大小
        # 4. 随机选择剪切位置
        # 5. 将 image2 的区域粘贴到 image1
        # 6. 按面积比例计算混合标签
        # 
        # 示例代码框架:
        # if np.random.random() < self.probability:
        #     lam = np.random.beta(self.alpha, self.alpha)
        #     
        #     # 获取随机矩形框
        #     H, W = image1.shape[:2]
        #     x1, y1, x2, y2 = self._get_rand_bbox(W, H, lam)
        #     
        #     # 剪切粘贴
        #     mixed_image = image1.copy()
        #     mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        #     
        #     # 计算实际混合比例（按面积）
        #     lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        #     
        #     # 混合标签
        #     num_classes = 5
        #     mixed_label = np.zeros(num_classes)
        #     mixed_label[label1] = lam
        #     mixed_label[label2] += (1 - lam)
        #     
        #     return mixed_image, mixed_label
        # ============================================================
        
        # 占位实现 - 直接返回原图
        print("[WARNING] CutMixAugmentation 尚未实现，请组员C完成")
        num_classes = 5
        one_hot = np.zeros(num_classes)
        one_hot[label1] = 1.0
        return image1, one_hot
    
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
        
        TODO: 组员C实现此方法
        提示:
        - 剪切区域面积 = (1 - lam) * 总面积
        - 随机选择中心点
        - 计算矩形边界
        """
        # TODO: 实现随机矩形框生成逻辑
        return 0, 0, width // 2, height // 2
    
    def get_config(self) -> dict:
        """获取增强配置"""
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config
