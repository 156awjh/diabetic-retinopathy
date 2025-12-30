"""
训练模块
========

提供模型训练相关功能。
"""

from .trainer import ModelTrainer
from .callbacks import get_callbacks

__all__ = ['ModelTrainer', 'get_callbacks']
