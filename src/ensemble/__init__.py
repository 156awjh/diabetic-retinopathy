"""
集成学习模块
============

提供模型集成和投票策略。
"""

from .ensemble_model import EnsembleModel
from .voting import hard_voting, soft_voting, weighted_voting

__all__ = ['EnsembleModel', 'hard_voting', 'soft_voting', 'weighted_voting']
