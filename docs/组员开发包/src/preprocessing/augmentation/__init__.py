"""
数据增强模块
============

本模块包含数据增强的基类和具体实现。

组员分工:
- geometric.py: 几何变换（组员A）- 旋转、翻转、裁剪
- color.py: 颜色变换（组员B）- 亮度、对比度、色调
- advanced.py: 高级增强（组员C）- Mixup 或 CutMix
"""

from .base import BaseAugmentation

__all__ = ['BaseAugmentation']
